import os
import glob
import re
import types
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
from einops import rearrange
import timm
import timm.data
from timm.models.vision_transformer import VisionTransformer
from model import SatUp
from tqdm import tqdm


# =========================
# timm 特征提取器（与原版 JAFAR 的 PretrainedViTWrapper 一致）
# =========================
class TimmViTFeature(nn.Module):
    def __init__(
        self, model_name="vit_base_patch16_clip_384", device="cuda", norm=True
    ):
        super().__init__()
        self.model_name = model_name
        self.norm = norm
        # 加载模型（无分类头，动态尺寸）
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            dynamic_img_size=True,
        )
        self.model = self.model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad = False

        # 获取数据配置（包括归一化参数和输入尺寸）
        self.data_config = timm.data.resolve_model_data_config(self.model)
        self.mean = self.data_config["mean"]
        self.std = self.data_config["std"]
        self.input_size = self.data_config["input_size"][-1]  # 通常为 384

        # 嵌入维度
        self.embed_dim = self.model.embed_dim  # 768

    @torch.no_grad()
    def forward(self, x):
        """
        输入: x (B, 3, H, W)，已经归一化
        输出: (B, embed_dim, H_patch, W_patch) 最后一层 patch 特征图
        """
        # 使用 forward_intermediates 提取最后一层 (n=1)
        # 注意：新版 timm 可能使用 indices 参数，此处兼容两种写法
        try:
            # 新版 timm (>=0.9.0) 使用 indices
            result = self.model.forward_intermediates(
                x,
                indices=[-1],  # 取最后一层
                return_prefix_tokens=True,
                norm=self.norm,
                output_fmt="NCHW",
                intermediates_only=False,
            )
        except TypeError:
            # 旧版 timm 使用 n
            result = self.model.forward_intermediates(
                x,
                n=1,
                return_prefix_tokens=True,
                norm=self.norm,
                output_fmt="NCHW",
                intermediates_only=False,
            )
        # 解析返回值：可能是 (feats, cls_token) 或只有 feats
        if isinstance(result, tuple):
            feats, cls_token = result
        else:
            feats = result
            cls_token = None
        # feats 可能是列表（因为指定了 indices/n），取第一个
        if isinstance(feats, list):
            feats = feats[0]
        # 如果 output_fmt="NCHW" 未生效，可能返回 (B, L, C)，手动重塑
        if feats.dim() == 3:
            B, L, C = feats.shape
            patch_tokens = feats[:, 1:, :]  # 去掉 CLS
            H = W = int((L - 1) ** 0.5)
            feats = patch_tokens.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return feats


# =========================
# Loss (JAFAR standard)
# =========================
class Cosine_MSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.cos = nn.CosineEmbeddingLoss()

    def forward(self, pred, target):
        pred = rearrange(pred, "b c h w -> (b h w) c")
        target = rearrange(target, "b c h w -> (b h w) c")
        y = torch.ones(pred.size(0), device=pred.device)
        return self.cos(pred, target, y) + self.mse(pred, target)


# =========================
# Dataset (HR only) – 使用 timm 的归一化参数
# =========================
class ImageFolderDataset(Dataset):
    def __init__(self, root, mean, std, size=384):
        self.files = sorted(glob.glob(os.path.join(root, "*.jpg")))
        self.tf = T.Compose(
            [
                T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC),
                T.RandomCrop(size),  # 或保持尺寸，这里用 resize 确保尺寸
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return self.tf(Image.open(self.files[idx]).convert("RGB"))


# =========================
# TRAIN (JAFAR PIPELINE – timm 版本)
# =========================
def train(
    data_root, epochs=50, batch_size=4, lr=2e-4, model_name="vit_base_patch16_clip_384"
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 初始化特征提取器（先获取归一化参数，以便构建 Dataset）
    # 这里先临时创建一个实例以获取 mean/std，但我们后续会复用
    tmp_extractor = TimmViTFeature(model_name=model_name, device=device)
    mean = tmp_extractor.mean
    std = tmp_extractor.std
    size = tmp_extractor.input_size

    # 构建数据集和数据加载器
    dataset = ImageFolderDataset(data_root, mean, std, size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # 创建模型和特征提取器（正式使用）
    model = SatUp(dim=128, v_dim=tmp_extractor.embed_dim).to(device)
    clip_extractor = TimmViTFeature(model_name=model_name, device=device)

    # 优化器和损失
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = Cosine_MSE().to(device)

    for epoch in range(epochs):
        total = 0
        pbar = tqdm(loader)

        for hr in pbar:
            hr = hr.to(device)

            # 1. HR feature (target)
            with torch.no_grad():
                hr_feat = clip_extractor(hr)

            # 2. 生成固定 0.5 倍下采样图 → Query
            query_img = F.interpolate(
                hr, scale_factor=0.5, mode="bicubic", align_corners=False
            )
            _, _, qh, qw = query_img.shape
            qh = (qh // 16) * 16
            qw = (qw // 16) * 16
            if qh > 0 and qw > 0:
                query_img = F.interpolate(query_img, size=(qh, qw), mode="bicubic")

            # 3. 生成随机低分辨率图 (0.25~0.5) → 用于提取 LR 特征 (KV)
            scale = np.random.uniform(0.25, 0.5)
            kv_img = F.interpolate(
                hr, scale_factor=scale, mode="bicubic", align_corners=False
            )
            _, _, kh, kw = kv_img.shape
            kh = (kh // 16) * 16
            kw = (kw // 16) * 16
            if kh > 0 and kw > 0:
                kv_img = F.interpolate(kv_img, size=(kh, kw), mode="bicubic")

            # 4. 提取 LR 特征 (Key/Value)
            with torch.no_grad():
                lr_feat = clip_extractor(kv_img)

            # 5. JAFAR 前向
            pred = model(
                image=query_img,
                features=lr_feat,
                output_size=hr_feat.shape[-2:],
            )

            loss = loss_fn(pred, hr_feat)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()
            pbar.set_postfix(loss=loss.item())

        print(f"Epoch {epoch}: {total / len(loader):.4f}")
        torch.save(model.state_dict(), f"satup_{epoch}.pth")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--model_name", type=str, default="vit_base_patch16_clip_384")

    args = parser.parse_args()

    train(
        data_root=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        model_name=args.model_name,
    )
