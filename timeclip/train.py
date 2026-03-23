import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pandas as pd
import cv2  # 必须依赖：用于高效绘图
from typing import Tuple, Dict, Any, Optional

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
# 2. 数据预处理 (保留 CV2 高效逻辑)
# ============================================================================

def sliding_window_norm(x: torch.Tensor, window: int = 64) -> torch.Tensor:
    if x.dim() == 1: x = x.unsqueeze(0)
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
                         size: Tuple[int, int] = (224, 224)) -> torch.Tensor:
    """使用 OpenCV 高效绘制时间序列图像"""
    x_norm = torch.clamp(sliding_window_norm(x, window), 0, 1)
    B, T = x_norm.shape
    H, W = size
    
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), 
        (255, 255, 0), (255, 0, 255), (0, 255, 255)
    ]
    c = colors[color % len(colors)]
    
    device = x_norm.device
    images = torch.zeros(B, 3, H, W, device=device)
    
    for b in range(B):
        ts_vals = x_norm[b].cpu().detach().numpy()
        img_np = np.zeros((H, W, 3), dtype=np.uint8)
        
        x_coords = np.linspace(0, W - 1, T).astype(int)
        y_coords = (H - 1 - ts_vals * (H - 1)).astype(int)
        
        # 使用 CV2 快速绘制折线
        for i in range(T - 1):
            cv2.line(img_np, (x_coords[i], y_coords[i]), 
                     (x_coords[i+1], y_coords[i+1]), c, 2)
        
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        images[b] = img_tensor.to(device)
        
    return images

class TimesCLIPDataset(Dataset):
    def __init__(self, data_tensor: torch.Tensor, seq_len: int = 336, pred_len: int = 96):
        self.data = data_tensor.squeeze(0)
        self.seq_len, self.pred_len = seq_len, pred_len
        T = self.data.shape[1]
        self.num_samples = max(0, T - seq_len - pred_len + 1)
        if self.num_samples <= 0: raise ValueError("数据长度不足")

    def __len__(self): return self.num_samples

    def __getitem__(self, idx):
        end_idx = idx + self.seq_len
        input_data = self.data[:, idx:end_idx]
        target_data = self.data[:, end_idx:end_idx + self.pred_len]
        return input_data, target_data

def load_m4_yearly_data(file_path: str = "./m4-yearly.csv"):
    if not os.path.exists(file_path): raise FileNotFoundError(f"文件不存在: {file_path}")
    df = pd.read_csv(file_path).select_dtypes(include=[np.number]).fillna(0)
    data_values = df.values.astype('float32')
    if len(data_values.shape) == 1: data_values = data_values.reshape(1, -1)
    else: data_values = data_values.T
    
    mean = np.mean(data_values, axis=1, keepdims=True)
    std = np.std(data_values, axis=1, keepdims=True)
    std = np.where(std == 0, 1, std)
    data_values = (data_values - mean) / std
    
    data_tensor = torch.from_numpy(data_values).unsqueeze(0)
    total_length = data_tensor.shape[2]
    train_len, val_len = int(total_length * 0.7), int(total_length * 0.15)
    
    return (data_tensor[:, :, :train_len], 
            data_tensor[:, :, train_len:train_len+val_len], 
            data_tensor[:, :, train_len+val_len:])

# ============================================================================
# 3. 模型组件
# ============================================================================

class MultiVariateVisionEncoder(nn.Module):
    # 修改处：直接接收已加载的 clip_model
    def __init__(self, clip_model, projector_dim: int = 512, freeze_vit: bool = True):
        super().__init__()
        self.model = clip_model
        
        self.freeze_vit = freeze_vit
        if freeze_vit:
            for param in self.model.visual.parameters(): param.requires_grad = False
        
        vit_hidden_size = self.model.visual.proj.shape[1]
        self.projector = nn.Linear(vit_hidden_size, projector_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        B, N, C, H, W = images.shape
        images_reshaped = images.view(B * N, C, H, W)
        
        with torch.set_grad_enabled(not self.freeze_vit):
            vision_features = self.model.encode_image(images_reshaped)
        
        return self.projector(vision_features).view(B, N, -1)

class MultiVariateLanguageEncoder(nn.Module):
    # 修改处：直接接收已加载的 clip_model
    def __init__(self, clip_model, projector_dim: int = 512, patch_len: int = 16, freeze_text: bool = True):
        super().__init__()
        self.model = clip_model
        
        text_hidden_size = self.model.text_projection.shape[1]
        self.text_input_proj = nn.Linear(patch_len, text_hidden_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, text_hidden_size) * 0.02)
        self.positional_embedding = nn.Parameter(torch.randn(1, 512, text_hidden_size))
        
        self.freeze_text = freeze_text
        if freeze_text:
            for param in self.model.text.parameters(): param.requires_grad = False
            
        self.projector = nn.Linear(text_hidden_size, projector_dim)
        self.patch_len = patch_len
        self.text_hidden_size = text_hidden_size

    def forward(self, patches: torch.Tensor):
        device = next(self.parameters()).device
        patches = patches.to(device)
        B, N, M, _ = patches.shape
        
        patches_reshaped = patches.view(B * N, M, self.patch_len)
        text_embeddings = self.text_input_proj(patches_reshaped)
        
        cls_tokens = self.cls_token.expand(B * N, -1, -1)
        embeddings = torch.cat([cls_tokens, text_embeddings], dim=1) + self.positional_embedding[:, :M+1, :]
        
        x = self.model.text.transformer(embeddings)
        cls_features = self.projector(x[:, 0, :]).view(B, N, -1)
        return cls_features, x.view(B, N, M+1, self.text_hidden_size)

class VariateSelector(nn.Module):
    def __init__(self, d_model: int = 512, nhead: int = 8, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
            dim_feedforward=d_model*4, dropout=dropout, activation='gelu', batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.query_projection = nn.Linear(d_model, d_model)

    def forward(self, vision_features, text_features):
        B, N, D = vision_features.shape
        combined = torch.cat([vision_features.unsqueeze(2), text_features.unsqueeze(2)], dim=2).view(B, N * 2, D)
        selected = self.transformer_encoder(combined).view(B, N, 2, D)
        return selected[:, :, 0, :], selected[:, :, 1, :], torch.ones(B, N, device=vision_features.device)

class ForecastGenerator(nn.Module):
    def __init__(self, d_model: int = 512, pred_len: int = 96):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.ReLU(), nn.Dropout(0.1),
            nn.LayerNorm(d_model), nn.Linear(d_model * 2, pred_len)
        )
    def forward(self, features): return self.generator(features)

# ============================================================================
# 4. 主模型 TimesCLIP
# ============================================================================

class TimesCLIP(nn.Module):
    def __init__(self, seq_len, pred_len, patch_len, d_model=512, nhead=8, num_layers=6,
                 dropout=0.1, use_variate_selector=True, vit_model_name="ViT-B-32",
                 vit_pretrained="laion2b_s34b_b79k", device=None):
        super().__init__()
        self.seq_len, self.pred_len, self.patch_len = seq_len, pred_len, patch_len
        self.use_variate_selector = use_variate_selector
        
        # 修改处：在这里统一加载一次 open_clip 模型
        import open_clip
        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            vit_model_name, pretrained=vit_pretrained, device=device)
        
        # 修改处：将相同的 clip_model 实例传递给视觉和文本编码器
        self.vision_encoder = MultiVariateVisionEncoder(self.clip_model, d_model, freeze_vit=False)
        self.language_encoder = MultiVariateLanguageEncoder(self.clip_model, d_model, patch_len, freeze_text=False)
        
        if use_variate_selector:
            self.variate_selector = VariateSelector(d_model, nhead, 2, dropout)
            
        self.fusion_layer = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.ReLU(), nn.Dropout(dropout), nn.LayerNorm(d_model))
        self.forecast_generator = ForecastGenerator(d_model, pred_len)

    def forward(self, x):
        B, N, L = x.shape
        device = next(self.parameters()).device
        x = x.to(device)
        
        # Vision Path (使用 CV2 绘图)
        images = time_series_to_image(x.view(B * N, L), color=0, window=64, size=(224, 224))
        vision_features = self.vision_encoder(images.view(B, N, 3, 224, 224))
        
        # Language Path
        M = L // self.patch_len
        patches = x[:, :, :M * self.patch_len].view(B, N, M, self.patch_len)
        text_features, _ = self.language_encoder(patches)
        
        # Selection & Fusion
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
# 5. 训练引擎 (简化日志)
# ============================================================================

class TimesCLIPTrainer:
    def __init__(self, model, train_loader, val_loader, test_loader, device):
        self.model = model.to(device)
        self.loaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
        self.device = device
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=3, factor=0.5)
        self.save_dir = "./checkpoints"
        os.makedirs(self.save_dir, exist_ok=True)
        self.best_val_loss = float('inf')
        self.best_path = ""

    def run_epoch(self, mode='train'):
        is_train = mode == 'train'
        self.model.train(is_train)
        total_loss, total_fore, total_cont = 0, 0, 0
        all_preds, all_targets = [], []
        
        loader = self.loaders[mode]
        # 仅在训练时显示进度条，评估时隐藏以减少日志
        iterator = tqdm(loader, desc=f"{mode}", disable=not is_train, leave=False)

        for data, targets in iterator:
            data, targets = data.to(self.device), targets.to(self.device)
            
            if is_train:
                self.optimizer.zero_grad()
                loss_dict = self.model.compute_loss(data, targets)
                loss = loss_dict["total_loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                # 训练时更新进度条
                iterator.set_postfix({'loss': f'{loss.item():.4f}'})
            else:
                with torch.no_grad():
                    loss_dict = self.model.compute_loss(data, targets)
                    output = self.model(data)
                    all_preds.append(output["predictions"].cpu())
                    all_targets.append(targets.cpu())
            
            total_loss += loss_dict["total_loss"].item()
            total_fore += loss_dict["forecast_loss"].item()
            total_cont += loss_dict["contrastive_loss"].item()

        n = len(loader)
        metrics = {"loss": total_loss/n, "fore": total_fore/n, "cont": total_cont/n}
        if not is_train:
            metrics.update(calculate_metrics(torch.cat(all_preds), torch.cat(all_targets)))
        return metrics

    def train(self, epochs=50, patience=8):
        print("开始训练...")
        wait = 0
        for epoch in range(epochs):
            train_m = self.run_epoch('train')
            val_m = self.run_epoch('val')
            
            self.scheduler.step(val_m["loss"])
            
            # 只打印关键信息
            print(f"Epoch {epoch+1}: Train Loss={train_m['loss']:.4f} | Val Loss={val_m['loss']:.4f} (MSE={val_m['mse']:.4f})")
            
            if val_m["loss"] < self.best_val_loss:
                self.best_val_loss = val_m["loss"]
                self.best_path = os.path.join(self.save_dir, f"best_epoch_{epoch+1}.pth")
                torch.save({'model_state_dict': self.model.state_dict(), 'val_loss': val_m['loss']}, self.best_path)
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"早停于 Epoch {epoch+1}")
                    break
        
        # 测试
        if os.path.exists(self.best_path):
            self.model.load_state_dict(torch.load(self.best_path, map_location=self.device)['model_state_dict'])
        test_m = self.run_epoch('test')
        print(f"\n=== 测试结果 ===")
        print(f"MSE: {test_m['mse']:.4f}, MAE: {test_m['mae']:.4f}, sMAPE: {test_m['smape']:.4f}")
        return test_m

# ============================================================================
# 6. 主入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--data_path', type=str, default="./timeclip/m4-yearly.csv")
    args = parser.parse_args()
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    cfg = {"seq_len": 96, "pred_len": 24, "patch_len": 12, "batch_size": 2, "d_model": 256}
    
    try:
        train_d, val_d, test_d = load_m4_yearly_data(args.data_path)
    except Exception as e:
        print(f"数据加载失败: {e}")
        return

    loaders = [DataLoader(TimesCLIPDataset(d, cfg["seq_len"], cfg["pred_len"]), 
                          batch_size=cfg["batch_size"], shuffle=(i==0), num_workers=0) 
               for i, d in enumerate([train_d, val_d, test_d])]
    
    model = TimesCLIP(seq_len=cfg["seq_len"], pred_len=cfg["pred_len"], patch_len=cfg["patch_len"],
                      d_model=cfg["d_model"], device=device)
    
    trainer = TimesCLIPTrainer(model, *loaders, device)
    trainer.train(epochs=50, patience=8)

if __name__ == "__main__":
    main()