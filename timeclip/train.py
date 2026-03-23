"""
TimesCLIP 完整训练脚本 - 单文件版本
整合了模型定义、数据加载、训练引擎和配置管理
用于时间序列预测的多模态对比学习方法
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
import argparse
import os
import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pandas as pd


# ============================================================================
# 第一部分：工具函数和指标计算
# ============================================================================

def mse(predictions, targets):
    """计算均方误差"""
    return ((predictions - targets) ** 2).mean().item()

def mae(predictions, targets):
    """计算平均绝对误差"""
    return (torch.abs(predictions - targets)).mean().item()

def smape(predictions, targets):
    """计算对称平均绝对百分比误差"""
    denominator = (torch.abs(targets) + torch.abs(predictions)) / 2
    smape_val = torch.abs(targets - predictions) / (denominator + 1e-5)
    return smape_val.mean().item()

def calculate_all_metrics(predictions, targets):
    """计算所有评估指标"""
    return {
        "mse": mse(predictions, targets),
        "mae": mae(predictions, targets),
        "smape": smape(predictions, targets)
    }


# ============================================================================
# 第二部分：对比学习损失
# ============================================================================

def compute_info_nce_loss(vision_features, text_features, mode="bidirectional", temperature=0.07):
    """
    计算 InfoNCE 对比损失
    
    Args:
        vision_features: 视觉特征 [B, N, D]
        text_features: 文本特征 [B, N, D]
        mode: 损失模式 ("bidirectional" or "unified")
        temperature: 温度参数
    
    Returns:
        contrastive_loss: 标量损失值
    """
    B, N, D = vision_features.shape
    
    # 重塑为 [B*N, D]
    vision_flat = vision_features.reshape(B * N, D)
    text_flat = text_features.reshape(B * N, D)
    
    # L2 归一化
    vision_flat = F.normalize(vision_flat, p=2, dim=-1)
    text_flat = F.normalize(text_flat, p=2, dim=-1)
    
    # 计算相似度矩阵
    similarity = torch.matmul(vision_flat, text_flat.T) / temperature  # [B*N, B*N]
    
    # 创建标签（对角线为正样本）
    labels = torch.arange(B * N, device=vision_flat.device)
    
    # 交叉熵损失
    loss_i = F.cross_entropy(similarity, labels)
    loss_t = F.cross_entropy(similarity.T, labels)
    
    if mode == "bidirectional":
        contrastive_loss = (loss_i + loss_t) / 2
    else:
        contrastive_loss = loss_i
    
    return contrastive_loss


# ============================================================================
# 第三部分：数据预处理模块
# ============================================================================

def sliding_window_norm(x: torch.Tensor, window: int = 64) -> torch.Tensor:
    """
    滑动窗口归一化
    
    Args:
        x: 时间序列张量 [B, T] 或 [T]
        window: 滑动窗口大小
    
    Returns:
        归一化后的张量
    """
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
                        size: Tuple[int, int] = (224, 224),
                        use_fast_method: bool = True) -> torch.Tensor:
    """
    将时间序列转换为 RGB 图像
    
    Args:
        x: 时间序列张量 [B, T]
        color: 颜色索引 (0-5)
        window: 滑动窗口大小
        size: 输出图像尺寸 (H, W)
        use_fast_method: 是否使用快速方法
    
    Returns:
        RGB 图像张量 [B, 3, H, W]
    """
    try:
        import cv2
        use_cv2 = True
    except ImportError:
        use_cv2 = False
    
    # 归一化
    x_norm = sliding_window_norm(x, window)
    
    B, T = x_norm.shape
    H, W = size
    
    # 颜色调色板
    colors = [
        (255, 0, 0),    # 红色
        (0, 255, 0),    # 绿色
        (0, 0, 255),    # 蓝色
        (255, 255, 0),  # 黄色
        (255, 0, 255),  # 洋红色
        (0, 255, 255),  # 青色
    ]
    
    # 限制值到 [0, 1]
    x_norm = torch.clamp(x_norm, 0, 1)
    
    # 创建空白 RGB 图像
    device = x_norm.device
    images = torch.zeros(B, 3, H, W, device=device)
    
    for b in range(B):
        if use_cv2 and use_fast_method:
            # 使用 OpenCV 快速生成
            time_series_values = x_norm[b].cpu().detach().numpy()
            img_np = np.zeros((H, W, 3), dtype=np.uint8)
            
            x_coords = np.linspace(0, W - 1, T).astype(int)
            y_coords = (H - 1 - time_series_values * (H - 1)).astype(int)
            
            for i in range(T - 1):
                cv2.line(img_np, 
                        (x_coords[i], y_coords[i]), 
                        (x_coords[i+1], y_coords[i+1]), 
                        colors[color], 2)
            
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.to(device)
            images[b] = img_tensor
        else:
            # 简单方法：直接映射
            time_series_values = x_norm[b].cpu().detach().numpy()
            img_np = np.zeros((H, W, 3), dtype=np.float32)
            
            for w in range(W):
                idx = min(int((w / W) * T), T - 1)
                value = time_series_values[idx]
                img_np[:, w, :] = np.array(colors[color]) * value / 255.0
            
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()
            img_tensor = img_tensor.to(device)
            images[b] = img_tensor
    
    return images


# ============================================================================
# 第四部分：数据集类
# ============================================================================

class TimesCLIPDataset(Dataset):
    """
    TimesCLIP 数据集，处理滑动窗口采样
    """
    def __init__(self, data_tensor: torch.Tensor, seq_len: int = 336, pred_len: int = 96):
        """
        Args:
            data_tensor: 时间序列数据张量 [1, N, T]
            seq_len: 输入序列长度
            pred_len: 预测长度
        """
        self.data = data_tensor.squeeze(0)  # [N, T]
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        T = self.data.shape[1]
        self.num_samples = T - seq_len - pred_len + 1
        
        if self.num_samples <= 0:
            raise ValueError(f"数据不足以进行滑动窗口采样。T={T}, seq_len={seq_len}, pred_len={pred_len}")
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        start_idx = idx
        end_idx = idx + self.seq_len
        target_start_idx = end_idx
        target_end_idx = target_start_idx + self.pred_len
        
        input_data = self.data[:, start_idx:end_idx]  # [N, seq_len]
        target_data = self.data[:, target_start_idx:target_end_idx]  # [N, pred_len]
        
        return input_data, target_data


def load_raw_data(dataset_name: str, data_type: str = "train", data_file: str = None) -> torch.Tensor:
    """
    加载原始时间序列数据
    
    Args:
        dataset_name: 数据集名称
        data_type: 数据类型 (train/val/test)
        data_file: 具体的 CSV 文件名
    
    Returns:
        时间序列数据张量 [1, N, T]
    """
    if dataset_name.startswith("ett"):
        data_path = os.path.join("data", "raw", "ett", data_type)
    else:
        data_path = os.path.join("data", "raw", dataset_name, data_type)
    
    if not os.path.exists(data_path):
        if data_type == "val":
            print(f"验证数据不存在，使用测试数据代替")
            data_path = os.path.join("data", "raw", "ett", "test")
        else:
            print(f"数据路径不存在：{data_path}")
            return None
    
    csv_files = sorted([f for f in os.listdir(data_path) if f.endswith('.csv')])
    
    if not csv_files:
        print(f"目录中没有 CSV 文件：{data_path}")
        return None
    
    if data_file and data_file in csv_files:
        csv_file = data_file
    else:
        csv_file = csv_files[0]
    
    file_path = os.path.join(data_path, csv_file)
    print(f"加载数据文件：{file_path}")
    
    df = pd.read_csv(file_path)
    data_values = df.iloc[:, 1:].values.astype('float32')
    data_values = data_values.T
    data_tensor = torch.from_numpy(data_values).unsqueeze(0)
    
    print(f"加载数据完成，形状：{data_tensor.shape}")
    return data_tensor


# ============================================================================
# 第五部分：模型组件
# ============================================================================

class MultiVariateVisionEncoder(nn.Module):
    """
    多变量视觉编码器，基于 CLIP ViT
    """
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32",
                 projector_dim: int = 512, freeze_vit: bool = True,
                 device: Optional[torch.device] = None):
        super().__init__()
        try:
            from transformers import CLIPVisionModel
            self.vit = CLIPVisionModel.from_pretrained(model_name)
        except Exception as e:
            print(f"加载 CLIP ViT 模型失败：{e}")
            print("请使用本地模型路径或检查网络连接")
            raise
        
        if freeze_vit:
            for param in self.vit.parameters():
                param.requires_grad = False
        
        vit_hidden_size = self.vit.config.hidden_size
        self.projector = nn.Linear(vit_hidden_size, projector_dim)
        self.projector_dim = projector_dim
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        B, N, C, H, W = images.shape
        
        images_reshaped = images.view(B * N, C, H, W)
        
        with torch.set_grad_enabled(self.vit.training):
            vit_outputs = self.vit(images_reshaped, output_hidden_states=False)
        
        cls_token = vit_outputs.last_hidden_state[:, 0, :]
        projected_cls = self.projector(cls_token)
        
        features = projected_cls.view(B, N, self.projector_dim)
        return features


class MultiVariateLanguageEncoder(nn.Module):
    """
    多变量语言编码器，基于 CLIP Text Model 改造
    """
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32",
                 projector_dim: int = 512, patch_len: int = 16,
                 fine_tune_layers: int = 2, freeze_text: bool = True):
        super().__init__()
        
        try:
            from transformers import CLIPTextModel
            self.text_encoder = CLIPTextModel.from_pretrained(model_name)
        except Exception as e:
            print(f"加载 CLIP Text 模型失败：{e}")
            print("请使用本地模型路径或检查网络连接")
            raise
        
        # 替换 token embedding 层
        self.text_encoder.text_model.embeddings.token_embedding = nn.Linear(patch_len, projector_dim)
        
        # 添加可学习的 CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, projector_dim) * 0.02)
        
        # 添加可学习的位置编码
        max_seq_len = 512
        self.positional_embedding = nn.Parameter(torch.randn(1, max_seq_len, projector_dim))
        
        # 冻结参数
        if freeze_text:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        
        if fine_tune_layers > 0:
            total_layers = len(self.text_encoder.text_model.encoder.layers)
            for i in range(total_layers - fine_tune_layers, total_layers):
                for param in self.text_encoder.text_model.encoder.layers[i].parameters():
                    param.requires_grad = True
        
        self.projector_dim = projector_dim
        self.patch_len = patch_len
    
    def forward(self, patches: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        patches = patches.to(device)
        
        B, N, M, _ = patches.shape
        
        patches_reshaped = patches.view(B * N, M, self.patch_len)
        
        token_embeddings = self.text_encoder.text_model.embeddings.token_embedding(patches_reshaped)
        
        cls_tokens = self.cls_token.expand(B * N, -1, -1)
        embeddings = torch.cat([cls_tokens, token_embeddings], dim=1)
        
        embeddings = embeddings + self.positional_embedding[:, :M+1, :]
        
        encoder_outputs = self.text_encoder.text_model.encoder(
            inputs_embeds=embeddings,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True
        )
        hidden_states = encoder_outputs.last_hidden_state
        
        hidden_states = self.text_encoder.text_model.final_layer_norm(hidden_states)
        
        cls_features = hidden_states[:, 0, :]
        
        cls_features = cls_features.view(B, N, self.projector_dim)
        all_features = hidden_states.view(B, N, M+1, self.projector_dim)
        
        return cls_features, all_features


class VariateSelector(nn.Module):
    """
    基于交叉注意力的变量选择器
    """
    def __init__(self, d_model: int = 512, nhead: int = 8, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.query_projection = nn.Linear(d_model, d_model)
        self.output_projection = nn.Linear(d_model * 2, d_model)
    
    def forward(self, vision_features: torch.Tensor, 
                text_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, D = vision_features.shape
        
        query = self.query_projection(text_features)
        
        combined = torch.cat([vision_features.unsqueeze(2), text_features.unsqueeze(2)], dim=2)
        combined = combined.view(B, N * 2, D)
        
        selected = self.transformer_encoder(combined)
        selected = selected.view(B, N, 2, D)
        
        selected_vision = selected[:, :, 0, :]
        selected_text = selected[:, :, 1, :]
        
        selection_weights = torch.ones(B, N, device=vision_features.device)
        
        return selected_vision, selected_text, selection_weights


class ForecastGenerator(nn.Module):
    """
    预测生成器
    """
    def __init__(self, d_model: int = 512, pred_len: int = 96, 
                 generator_type: str = "transformer"):
        super().__init__()
        
        if generator_type == "transformer":
            self.generator = nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.LayerNorm(d_model),
                nn.Linear(d_model * 2, pred_len)
            )
        else:
            self.generator = nn.Linear(d_model, pred_len)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        B, N, D = features.shape
        
        predictions = self.generator(features)
        
        return predictions


# ============================================================================
# 第六部分：TimesCLIP 主模型
# ============================================================================

class TimesCLIP(nn.Module):
    """
    TimesCLIP 主模型：结合视觉和语言编码器的多模态时间序列预测模型
    """
    def __init__(self, seq_len: int, pred_len: int, patch_len: int,
                 d_model: int = 512, nhead: int = 8, num_layers: int = 6,
                 dropout: float = 0.1, use_variate_selector: bool = True,
                 variate_selector_type: str = "cross_attention",
                 generator_type: str = "transformer",
                 contrastive_loss_mode: str = "bidirectional",
                 vit_model_name: str = "openai/clip-vit-base-patch32",
                 text_model_name: str = "openai/clip-vit-base-patch32",
                 device: Optional[torch.device] = None):
        
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.d_model = d_model
        self.use_variate_selector = use_variate_selector
        self.contrastive_loss_mode = contrastive_loss_mode
        
        # 视觉编码器
        self.vision_encoder = MultiVariateVisionEncoder(
            model_name=vit_model_name,
            projector_dim=d_model,
            freeze_vit=False,
            device=device
        )
        
        # 语言编码器
        self.language_encoder = MultiVariateLanguageEncoder(
            model_name=text_model_name,
            projector_dim=d_model,
            patch_len=patch_len,
            fine_tune_layers=2,
            freeze_text=False
        )
        
        # 变量选择器
        if use_variate_selector:
            if variate_selector_type == "cross_attention":
                self.variate_selector = VariateSelector(
                    d_model=d_model,
                    nhead=nhead,
                    num_layers=2,
                    dropout=dropout
                )
            else:
                raise ValueError(f"不支持的变量选择器类型：{variate_selector_type}")
        
        # 预测生成器
        self.forecast_generator = ForecastGenerator(
            d_model=d_model,
            pred_len=pred_len,
            generator_type=generator_type
        )
        
        # 特征融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model)
        )
    
    def forward(self, x: torch.Tensor, return_features: bool = False) -> dict:
        device = next(self.parameters()).device
        x = x.to(device)
        
        B, N, L = x.shape
        
        # 视觉通路：时序 → 图像 → ViT
        x_reshaped = x.view(B * N, L)
        images = time_series_to_image(x_reshaped, color=0, window=64, size=(224, 224))
        images = images.to(device)
        images = images.view(B, N, 3, 224, 224)
        
        vision_features = self.vision_encoder(images)
        
        # 语言通路：时序 → Patches → Text Encoder
        M = L // self.patch_len
        x_patches = x[:, :, :M * self.patch_len].view(B, N, M, self.patch_len)
        
        text_features, _ = self.language_encoder(x_patches)
        
        # 变量选择
        if self.use_variate_selector:
            selected_vision, selected_text, selection_weights = self.variate_selector(
                vision_features, text_features
            )
        else:
            selected_vision, selected_text = vision_features, text_features
            selection_weights = None
        
        # 特征融合
        fused_features = self.fusion_layer(
            torch.cat([selected_vision, selected_text], dim=-1)
        )
        
        # 生成预测
        predictions = self.forecast_generator(fused_features)
        
        output = {
            "predictions": predictions,
            "vision_features": vision_features,
            "text_features": text_features,
            "fused_features": fused_features
        }
        
        if self.use_variate_selector:
            output["selection_weights"] = selection_weights
        
        if return_features:
            output["all_features"] = {
                "vision": vision_features,
                "text": text_features,
                "fused": fused_features
            }
        
        return output
    
    def compute_loss(self, x: torch.Tensor, targets: torch.Tensor, 
                    contrastive_weight: float = 0.1) -> dict:
        output = self.forward(x)
        predictions = output["predictions"]
        vision_features = output["vision_features"]
        text_features = output["text_features"]
        
        forecast_loss = F.mse_loss(predictions, targets)
        
        contrastive_loss = compute_info_nce_loss(
            vision_features, 
            text_features, 
            mode=self.contrastive_loss_mode
        )
        
        weighted_contrastive_loss = contrastive_loss * contrastive_weight
        
        total_loss = forecast_loss + weighted_contrastive_loss
        
        return {
            "total_loss": total_loss,
            "forecast_loss": forecast_loss,
            "contrastive_loss": weighted_contrastive_loss,
            "unweighted_contrastive_loss": contrastive_loss
        }


# ============================================================================
# 第七部分：训练引擎
# ============================================================================

class TrainingLogger:
    """训练日志记录器"""
    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.log_dir, f"training_log_{timestamp}.json")
        
        self.log_data = {
            "start_time": datetime.now().isoformat(),
            "config": {},
            "epochs": []
        }
        self._write_log()
    
    def set_config(self, config: Dict[str, Any]):
        self.log_data["config"] = config
        self._write_log()
    
    def log_epoch(self, epoch: int, train_metrics: Dict[str, float], 
                  val_metrics: Dict[str, float]):
        epoch_data = {
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            "train": train_metrics,
            "validation": val_metrics
        }
        
        self.log_data["epochs"].append(epoch_data)
        self._write_log()
    
    def log_final_results(self, test_metrics: Dict[str, float], best_model_path: str):
        self.log_data["end_time"] = datetime.now().isoformat()
        self.log_data["final_test_metrics"] = test_metrics
        self.log_data["best_model_path"] = best_model_path
        self._write_log()
    
    def _write_log(self):
        try:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(self.log_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"写入日志失败：{e}")
    
    def get_log_path(self) -> str:
        return self.log_file


class TimesCLIPTrainer:
    """
    TimesCLIP 训练器
    """
    def __init__(self, model: nn.Module, train_loader: DataLoader,
                 val_loader: DataLoader, test_loader: DataLoader,
                 config: Dict[str, Any], device: torch.device):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device
        
        self.learning_rate = float(config.get("learning_rate", 1e-4))
        self.weight_decay = float(config.get("weight_decay", 1e-5))
        self.num_epochs = int(config.get("num_epochs", 100))
        self.patience = int(config.get("patience", 10))
        self.save_dir = config.get("save_dir", "./checkpoints")
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=3,
            factor=0.5,
            verbose=True
        )
        
        self.logger = self._setup_logger()
        os.makedirs(self.save_dir, exist_ok=True)
        
        log_dir = config.get("log_dir", "./logs")
        self.training_logger = TrainingLogger(log_dir)
        self.training_logger.set_config(config)
        
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
    
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("TimesCLIPTrainer")
        logger.setLevel(logging.INFO)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        ch.setFormatter(formatter)
        
        if not logger.handlers:
            logger.addHandler(ch)
        
        return logger
    
    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_forecast_loss = 0.0
        total_contrastive_loss = 0.0
        num_batches = len(self.train_loader)
        
        contrastive_weight = self.config.get("model", {}).get("contrastive_weight", 0.1)
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch in progress_bar:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                data, targets = batch
                data = data.to(self.device)
                targets = targets.to(self.device)
            else:
                continue
            
            self.optimizer.zero_grad()
            
            loss_dict = self.model.compute_loss(data, targets, contrastive_weight=contrastive_weight)
            loss = loss_dict["total_loss"]
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_forecast_loss += loss_dict["forecast_loss"].item()
            total_contrastive_loss += loss_dict["contrastive_loss"].item()
            
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        return {
            "loss": total_loss / num_batches,
            "forecast_loss": total_forecast_loss / num_batches,
            "contrastive_loss": total_contrastive_loss / num_batches
        }
    
    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_forecast_loss = 0.0
        total_contrastive_loss = 0.0
        all_predictions = []
        all_targets = []
        
        contrastive_weight = self.config.get("model", {}).get("contrastive_weight", 0.1)
        
        for batch in tqdm(data_loader, desc="Evaluating"):
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                data, targets = batch
                data = data.to(self.device)
                targets = targets.to(self.device)
            else:
                continue
            
            loss_dict = self.model.compute_loss(data, targets, contrastive_weight=contrastive_weight)
            
            total_loss += loss_dict["total_loss"].item()
            total_forecast_loss += loss_dict["forecast_loss"].item()
            total_contrastive_loss += loss_dict["contrastive_loss"].item()
            
            output = self.model(data)
            predictions = output["predictions"]
            
            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())
        
        num_batches = len(all_predictions)
        if num_batches == 0:
            num_batches = 1
        
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        mse_score = mse(all_predictions, all_targets)
        mae_score = mae(all_predictions, all_targets)
        smape_score = smape(all_predictions, all_targets)
        
        return {
            "loss": total_loss / num_batches,
            "forecast_loss": total_forecast_loss / num_batches,
            "contrastive_loss": total_contrastive_loss / num_batches,
            "mse": mse_score,
            "mae": mae_score,
            "smape": smape_score
        }
    
    def train(self) -> Dict[str, Any]:
        self.logger.info("开始训练...")
        
        history = {
            "train_losses": [],
            "val_losses": [],
            "val_metrics": []
        }
        
        for epoch in range(self.num_epochs):
            self.logger.info(f"Epoch {epoch+1}/{self.num_epochs}")
            
            train_metrics = self.train_epoch()
            history["train_losses"].append(train_metrics["loss"])
            
            val_metrics = self.evaluate(self.val_loader)
            history["val_losses"].append(val_metrics["loss"])
            history["val_metrics"].append(val_metrics)
            
            self.logger.info(
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"MSE: {val_metrics['mse']:.4f}"
            )
            
            self.training_logger.log_epoch(epoch+1, train_metrics, val_metrics)
            
            self.scheduler.step(val_metrics["loss"])
            
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.best_epoch = epoch
                self.patience_counter = 0
                
                dataset_name = self.config.get('data', {}).get('dataset_name', 'unknown')
                best_model_path = os.path.join(
                    self.save_dir, 
                    f"timesclip_{dataset_name}_best_epoch_{epoch+1}.pth"
                )
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_metrics["loss"],
                }, best_model_path)
                
                self.logger.info(f"保存最佳模型到 {best_model_path}")
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= self.patience:
                self.logger.info(f"早停于 epoch {epoch+1}")
                break
        
        self.logger.info("训练完成，在测试集上评估...")
        test_metrics = self.evaluate(self.test_loader)
        
        self.logger.info(
            f"Test Metrics - Loss: {test_metrics['loss']:.4f}, "
            f"MSE: {test_metrics['mse']:.4f}, "
            f"MAE: {test_metrics['mae']:.4f}"
        )
        
        best_model_path = os.path.join(
            self.save_dir, 
            f"timesclip_{self.config.get('data', {}).get('dataset_name', 'unknown')}_best_epoch_{self.best_epoch+1}.pth"
        )
        self.training_logger.log_final_results(test_metrics, best_model_path)
        
        return {
            "history": history,
            "test_metrics": test_metrics,
            "best_model_path": best_model_path,
            "log_file": self.training_logger.get_log_path()
        }


# ============================================================================
# 第八部分：主函数
# ============================================================================

def load_config(config_path: str) -> dict:
    """从 YAML 文件加载配置"""
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config


def create_timesclip_model(config: dict) -> TimesCLIP:
    """从配置创建 TimesCLIP 模型"""
    data_config = config.get("data", {})
    model_config = config.get("model", {})
    
    model = TimesCLIP(
        seq_len=data_config.get("seq_len", 512),
        pred_len=data_config.get("pred_len", 96),
        patch_len=data_config.get("patch_len", 16),
        d_model=model_config.get("d_model", 512),
        nhead=model_config.get("n_heads", 8),
        num_layers=model_config.get("num_layers", 6),
        dropout=model_config.get("dropout", 0.1),
        use_variate_selector=model_config.get("use_variate_selector", True),
        variate_selector_type=model_config.get("variate_selector_type", "cross_attention"),
        generator_type=model_config.get("generator_type", "transformer"),
        contrastive_loss_mode=model_config.get("contrastive_loss_mode", "bidirectional"),
        vit_model_name=model_config.get("vit_model", "openai/clip-vit-base-patch32"),
        text_model_name=model_config.get("text_model", "openai/clip-vit-base-patch32")
    )
    
    return model


def main():
    """主训练函数"""
    parser = argparse.ArgumentParser(description='Train TimesCLIP model')
    parser.add_argument('--config', type=str, required=True, 
                        help='Path to configuration file')
    parser.add_argument('--gpu', type=int, default=0, 
                        help='GPU device to use (default: 0)')
    parser.add_argument('--fast_mode', action='store_true',
                        help='Enable fast training mode')
    args = parser.parse_args()
    
    config = load_config(args.config)
    print(f"加载配置：{args.config}")
    
    if args.fast_mode:
        print("🚀 启用快速训练模式")
        if "training" not in config:
            config["training"] = {}
        
        current_max_steps = config["training"].get("max_steps", 500)
        if current_max_steps is not None:
            config["training"]["max_steps"] = min(current_max_steps, 100)
        else:
            config["training"]["max_steps"] = 100
            
        config["training"]["num_epochs"] = min(config["training"].get("num_epochs", 50), 10)
        config["training"]["batch_size"] = max(config["training"].get("batch_size", 32), 64)
        
        print(f"  最大步数：{config['training']['max_steps']}")
        print(f"  训练轮数：{config['training']['num_epochs']}")
        print(f"  批次大小：{config['training']['batch_size']}")
    
    if "dataset_name" in config["data"]:
        dataset_name = config["data"]["dataset_name"]
    elif "name" in config["data"]:
        dataset_name = config["data"]["name"]
    else:
        raise ValueError("配置中未找到数据集名称")
    
    data_file = config["data"].get("data_file", None)
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")
    
    print("\n加载数据...")
    train_data = load_raw_data(dataset_name, "train", data_file)
    if train_data is None:
        print("加载训练数据失败")
        return
    
    val_data = load_raw_data(dataset_name, "val", data_file)
    if val_data is None:
        print("验证数据不存在，使用训练数据代替")
        val_data = train_data
    
    test_data = load_raw_data(dataset_name, "test", data_file)
    if test_data is None:
        print("测试数据不存在，使用训练数据代替")
        test_data = train_data
    
    print(f"训练数据形状：{train_data.shape}")
    print(f"验证数据形状：{val_data.shape}")
    print(f"测试数据形状：{test_data.shape}")
    
    seq_len = config["data"].get("seq_len", 336)
    pred_len = config["data"].get("pred_len", 96)
    
    train_dataset = TimesCLIPDataset(train_data, seq_len, pred_len)
    val_dataset = TimesCLIPDataset(val_data, seq_len, pred_len)
    test_dataset = TimesCLIPDataset(test_data, seq_len, pred_len)
    
    batch_size = config["training"].get("batch_size", 32)
    num_workers = min(4, os.cpu_count() // 2) if os.name != 'nt' else 0
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\n数据集大小:")
    print(f"  训练集：{len(train_dataset)} 样本")
    print(f"  验证集：{len(val_dataset)} 样本")
    print(f"  测试集：{len(test_dataset)} 样本")
    
    print("\n创建模型...")
    model = create_timesclip_model(config)
    
    print("\n创建训练器...")
    trainer = TimesCLIPTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        device=device
    )
    
    print("\n开始训练...")
    results = trainer.train()
    
    print("\n" + "="*50)
    print("训练完成!")
    print("="*50)
    print(f"最佳验证损失：{results['test_metrics']['loss']:.4f}")
    print(f"测试 MSE: {results['test_metrics']['mse']:.4f}")
    print(f"测试 MAE: {results['test_metrics']['mae']:.4f}")
    print(f"测试 sMAPE: {results['test_metrics']['smape']:.4f}")
    print(f"最佳模型路径：{results['best_model_path']}")
    print(f"日志文件：{results['log_file']}")


if __name__ == "__main__":
    main()
