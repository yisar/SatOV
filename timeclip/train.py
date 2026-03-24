import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pandas as pd
import cv2
from typing import Tuple
import open_clip

# ============================================================================
# 1. 工具函数与指标
# ============================================================================
def calculate_metrics(predictions, targets):
    mse_val = ((predictions - targets) ** 2).mean().item()
    mae_val = torch.abs(predictions - targets).mean().item()
    denominator = (torch.abs(targets) + torch.abs(predictions)) / 2
    smape_val = (torch.abs(targets - predictions) / (denominator + 1e-5)).mean().item()
    return {"mse": mse_val, "mae": mae_val, "smape": smape_val}

def compute_info_nce_loss(vision_features, text_features, mode="bidirectional", temperature=0.07):
    B, N, D = vision_features.shape
    vision_flat = F.normalize(vision_features.reshape(B * N, D), p=2, dim=-1)
    text_flat = F.normalize(text_features.reshape(B * N, D), p=2, dim=-1)
    
    similarity = torch.matmul(vision_flat, text_flat.T) / temperature
    labels = torch.arange(B * N, device=vision_flat.device)
    
    loss_i = F.cross_entropy(similarity, labels)
    loss_t = F.cross_entropy(similarity.T, labels)
    return (loss_i + loss_t) / 2 if mode == "bidirectional" else loss_i

# ============================================================================
# 2. 数据预处理 (96x96像素)
# ============================================================================
def sliding_window_norm(x: torch.Tensor, window: int = 64) -> torch.Tensor:
    if x.dim() == 1:
        x = x.unsqueeze(0)
    B, T = x.shape
    normalized = torch.zeros_like(x)
    for i in range(0, T, window):
        end_idx = min(i + window, T)
        window_data = x[:, i:end_idx]
        min_vals = torch.min(window_data, dim=1, keepdim=True)[0]
        max_vals = torch.max(window_data, dim=1, keepdim=True)[0]
        range_vals = max_vals - min_vals
        range_vals = torch.where(range_vals == 0, torch.ones_like(range_vals), range_vals)
        normalized[:, i:end_idx] = (window_data - min_vals) / range_vals
    return normalized

def time_series_to_image(x: torch.Tensor, color: int = 0, window: int = 64,
                         size: Tuple[int, int] = (96, 96)) -> torch.Tensor:
    x_norm = torch.clamp(sliding_window_norm(x, window), 0, 1)
    B, T = x_norm.shape
    H, W = size
    
    colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255)]
    c = colors[color % len(colors)]
    
    device = x_norm.device
    images = torch.zeros(B, 3, H, W, device=device)
    
    for b in range(B):
        ts_vals = x_norm[b].detach().cpu().numpy()
        img_np = np.zeros((H, W, 3), dtype=np.uint8)
        
        x_coords = np.linspace(0, W-1, T).astype(int)
        y_coords = (H-1 - ts_vals * (H-1)).astype(int)
        
        for i in range(T-1):
            cv2.line(img_np, (x_coords[i], y_coords[i]), (x_coords[i+1], y_coords[i+1]), c, 2)
        
        img_tensor = torch.from_numpy(img_np).permute(2,0,1).float() / 255.0
        images[b] = img_tensor.to(device)
    return images

class TimesCLIPDataset(Dataset):
    def __init__(self, data_tensor: torch.Tensor, seq_len: int = 48, pred_len: int = 24):
        self.data = data_tensor.squeeze(0)
        self.seq_len, self.pred_len = seq_len, pred_len
        T = self.data.shape[1]
        self.num_samples = max(0, T - seq_len - pred_len + 1)
        if self.num_samples <= 0:
            raise ValueError("数据长度不足")

    def __len__(self): return self.num_samples
    def __getitem__(self, idx):
        end_idx = idx + self.seq_len
        input_data = self.data[:, idx:end_idx]
        target_data = self.data[:, end_idx:end_idx+self.pred_len]
        return input_data, target_data

def load_m4_yearly_data(file_path: str = "./m4-yearly.csv"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    df = pd.read_csv(file_path).select_dtypes(include=[np.number]).fillna(0)
    data_values = df.values.astype('float32')
    
    if len(data_values.shape) == 1:
        data_values = data_values.reshape(1, -1)
    else:
        data_values = data_values.T
    
    mean = np.mean(data_values, axis=1, keepdims=True)
    std = np.std(data_values, axis=1, keepdims=True)
    std = np.where(std == 0, 1, std)
    data_values = (data_values - mean) / std
    
    data_tensor = torch.from_numpy(data_values).unsqueeze(0)
    total_length = data_tensor.shape[2]
    train_len = int(total_length * 0.7)
    val_len = int(total_length * 0.15)
    
    return (data_tensor[:, :, :train_len],
            data_tensor[:, :, train_len:train_len+val_len],
            data_tensor[:, :, train_len+val_len:])

# ============================================================================
# 3. 模型组件（✅ 修复维度错误）
# ============================================================================
class MultiVariateVisionEncoder(nn.Module):
    def __init__(self, clip_model, projector_dim: int = 256, freeze_vit: bool = True):
        super().__init__()
        self.model = clip_model
        self.freeze_vit = freeze_vit
        
        if freeze_vit:
            for param in self.model.visual.parameters():
                param.requires_grad = False
        
        vit_hidden_size = self.model.visual.proj.shape[1]
        self.projector = nn.Linear(vit_hidden_size, projector_dim)
        self.patch_size = self.model.visual.patch_size[0]

    def interpolate_pos_embed(self, pos_embed, new_size):
        orig_size = int(np.sqrt(pos_embed.shape[0] - 1))
        new_h, new_w = new_size
        pos_embed_tok = pos_embed[0:1, :]
        pos_embed_img = pos_embed[1:, :].reshape(orig_size, orig_size, -1).permute(2, 0, 1)
        pos_embed_img = F.interpolate(pos_embed_img.unsqueeze(0), size=(new_h, new_w), mode='bicubic').squeeze(0)
        pos_embed_img = pos_embed_img.permute(1, 2, 0).reshape(new_h*new_w, -1)
        return torch.cat([pos_embed_tok, pos_embed_img], dim=0)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        B, N, C, H, W = images.shape
        images_reshaped = images.reshape(B*N, C, H, W)
        
        with torch.set_grad_enabled(not self.freeze_vit):
            num_patches = (H // self.patch_size) * (W // self.patch_size)
            orig_pos_emb = self.model.visual.positional_embedding
            if orig_pos_emb.shape[0] != num_patches + 1:
                new_pos_emb = self.interpolate_pos_embed(orig_pos_emb, (H//self.patch_size, W//self.patch_size))
                self.model.visual.positional_embedding = nn.Parameter(new_pos_emb)
            
            vision_features = self.model.encode_image(images_reshaped)
        
        return self.projector(vision_features).view(B, N, -1)

class MultiVariateLanguageEncoder(nn.Module):
    def __init__(self, clip_model, projector_dim: int = 256, patch_len: int = 12, freeze_text: bool = False):
        super().__init__()
        self.model = clip_model
        text_hidden_size = self.model.text_projection.shape[1]
        self.text_input_proj = nn.Linear(patch_len, text_hidden_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, text_hidden_size) * 0.02)
        self.pos_emb = nn.Parameter(torch.randn(1, 1, text_hidden_size))
        
        self.freeze_text = freeze_text
        if freeze_text:
            for param in self.model.transformer.parameters():
                param.requires_grad = False
        
        self.projector = nn.Linear(text_hidden_size, projector_dim)
        self.patch_len = patch_len

    def forward(self, patches: torch.Tensor):
        B, N, M, _ = patches.shape
        patches_reshaped = patches.reshape(B*N, M, self.patch_len)
        text_embeddings = self.text_input_proj(patches_reshaped)
        
        cls_tokens = self.cls_token.expand(B*N, -1, -1)
        embeddings = torch.cat([cls_tokens, text_embeddings], dim=1)
        
        seq_len = embeddings.shape[1]
        embeddings = embeddings + self.pos_emb.expand(-1, seq_len, -1)
        
        x = self.model.transformer(embeddings)
        cls_features = self.projector(x[:, 0, :]).view(B, N, -1)
        return cls_features, x

class VariateSelector(nn.Module):
    def __init__(self, d_model: int = 256, nhead: int = 8, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
            dim_feedforward=d_model*4, dropout=dropout, activation='gelu', batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, vision_features, text_features):
        B, N, D = vision_features.shape
        combined = torch.cat([vision_features.unsqueeze(2), text_features.unsqueeze(2)], dim=2).reshape(B, N*2, D)
        selected = self.transformer_encoder(combined).reshape(B, N, 2, D)
        return selected[:, :, 0, :], selected[:, :, 1, :], torch.ones(B, N, device=vision_features.device)

class ForecastGenerator(nn.Module):
    def __init__(self, d_model: int = 256, pred_len: int = 24):
        super().__init__()
        # ✅ 核心修复：LayerNorm 位置调整，维度匹配
        self.generator = nn.Sequential(
            nn.LayerNorm(d_model),  # 先归一化输入
            nn.Linear(d_model, d_model*2), 
            nn.ReLU(), 
            nn.Dropout(0.1),
            nn.Linear(d_model*2, pred_len)  # 最后直接输出
        )
    def forward(self, features): return self.generator(features)

# ============================================================================
# 4. 主模型
# ============================================================================
class TimesCLIP(nn.Module):
    def __init__(self, seq_len, pred_len, patch_len, d_model=256, nhead=8,
                 dropout=0.1, use_variate_selector=True, device=None):
        super().__init__()
        self.seq_len, self.pred_len, self.patch_len = seq_len, pred_len, patch_len
        self.use_variate_selector = use_variate_selector
        
        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k", device=device)
        
        self.vision_encoder = MultiVariateVisionEncoder(self.clip_model, d_model, freeze_vit=True)
        self.language_encoder = MultiVariateLanguageEncoder(self.clip_model, d_model, patch_len, freeze_text=False)
        
        if use_variate_selector:
            self.variate_selector = VariateSelector(d_model, nhead, 2, dropout)
        
        # ✅ 修复：融合层的 LayerNorm 维度也确认匹配
        self.fusion_layer = nn.Sequential(
            nn.Linear(d_model*2, d_model), 
            nn.ReLU(), 
            nn.Dropout(dropout), 
            nn.LayerNorm(d_model)  # 这里输入是 d_model，正确
        )
        self.forecast_generator = ForecastGenerator(d_model, pred_len)

    def forward(self, x):
        B, N, L = x.shape
        images = time_series_to_image(x.reshape(B*N, L), size=(96,96))
        vision_features = self.vision_encoder(images.reshape(B, N, 3, 96, 96))
        
        M = L // self.patch_len
        patches = x[:, :, :M*self.patch_len].reshape(B, N, M, self.patch_len)
        text_features, _ = self.language_encoder(patches)
        
        if self.use_variate_selector:
            sel_vis, sel_txt, _ = self.variate_selector(vision_features, text_features)
        else:
            sel_vis, sel_txt = vision_features, text_features
        
        fused = self.fusion_layer(torch.cat([sel_vis, sel_txt], dim=-1))
        return {"predictions": self.forecast_generator(fused), "vision_features": vision_features, "text_features": text_features}

    def compute_loss(self, x, targets, contrastive_weight=0.1):
        out = self.forward(x)
        forecast_loss = F.mse_loss(out["predictions"], targets)
        cont_loss = compute_info_nce_loss(out["vision_features"], out["text_features"])
        return {
            "total_loss": forecast_loss + cont_loss * contrastive_weight,
            "forecast_loss": forecast_loss,
            "contrastive_loss": cont_loss * contrastive_weight
        }

# ============================================================================
# 5. 训练引擎
# ============================================================================
class TimesCLIPTrainer:
    def __init__(self, model, train_loader, val_loader, test_loader, device):
        self.model = model.to(device)
        self.loaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=3, factor=0.5)
        self.save_dir = "./checkpoints"
        os.makedirs(self.save_dir, exist_ok=True)
        self.best_val_loss = float('inf')

    def run_epoch(self, mode='train'):
        self.model.train(mode == 'train')
        total_loss = 0
        all_preds, all_targets = [], []
        
        loader = self.loaders[mode]
        iterator = tqdm(loader, desc=mode, disable=mode!='train')
        
        for data, targets in iterator:
            data, targets = data.to(self.device), targets.to(self.device)
            
            if mode == 'train':
                self.optimizer.zero_grad()
                loss = self.model.compute_loss(data, targets)["total_loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                iterator.set_postfix(loss=f"{loss.item():.3f}")
            else:
                with torch.no_grad():
                    loss = self.model.compute_loss(data, targets)["total_loss"]
                    preds = self.model(data)["predictions"]
                    all_preds.append(preds.cpu())
                    all_targets.append(targets.cpu())
            
            total_loss += loss.item()
        
        metrics = {"loss": total_loss / len(loader)}
        if mode != 'train':
            metrics.update(calculate_metrics(torch.cat(all_preds), torch.cat(all_targets)))
        return metrics

    def train(self, epochs=50, patience=8):
        print("训练开始...")
        wait = 0
        for epoch in range(epochs):
            train_m = self.run_epoch('train')
            val_m = self.run_epoch('val')
            self.scheduler.step(val_m["loss"])
            
            print(f"Epoch {epoch+1} | Train Loss: {train_m['loss']:.4f} | Val Loss: {val_m['loss']:.4f} | Val MSE: {val_m['mse']:.4f}")
            
            if val_m["loss"] < self.best_val_loss:
                self.best_val_loss = val_m["loss"]
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, "best.pth"))
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print("早停触发")
                    break
        
        self.model.load_state_dict(torch.load(os.path.join(self.save_dir, "best.pth"), map_location=self.device))
        test_m = self.run_epoch('test')
        print(f"\n测试结果：MSE={test_m['mse']:.4f}, MAE={test_m['mae']:.4f}, sMAPE={test_m['smape']:.4f}")

# ============================================================================
# 主入口
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--data_path', type=str, default="./timeclip/m4-yearly.csv")
    args = parser.parse_args()
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    model_cfg = {
        "seq_len": 48,
        "pred_len": 24,
        "patch_len": 12,
        "d_model": 256
    }
    data_cfg = {"batch_size": 1}
    
    try:
        train_d, val_d, test_d = load_m4_yearly_data(args.data_path)
        print("✅ 数据加载成功")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return
    
    loaders = [
        DataLoader(TimesCLIPDataset(d, model_cfg["seq_len"], model_cfg["pred_len"]), 
                  batch_size=data_cfg["batch_size"], shuffle=i==0)
        for i, d in enumerate([train_d, val_d, test_d])
    ]
    
    model = TimesCLIP(**model_cfg, device=device)
    trainer = TimesCLIPTrainer(model, *loaders, device)
    trainer.train()

if __name__ == "__main__":
    main()