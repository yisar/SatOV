import os
import glob
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
from einops import rearrange
import open_clip
from model import SatUp  # 你的 JAFAR 模型
from tqdm import tqdm

# =========================
# 损失函数（移除错误归一化）
# =========================
class Cosine_MSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.cos = nn.CosineEmbeddingLoss()

    def forward(self, pred, target):
        # pred, target: (B, C, H, W)
        B, C, H, W = pred.shape
        pred_flat = rearrange(pred, 'b c h w -> (b h w) c')
        target_flat = rearrange(target, 'b c h w -> (b h w) c')
        # CosineEmbeddingLoss 需要标签：1 表示相似
        labels = torch.ones(pred_flat.size(0), device=pred.device)
        cos_loss = self.cos(pred_flat, target_flat, labels)
        mse_loss = self.mse(pred, target)
        return cos_loss + mse_loss

# =========================
# 数据集（保持不变）
# =========================
class ImageFolderDataset(Dataset):
    def __init__(self, root, patch_size=224, scale=2):
        self.files = sorted(glob.glob(os.path.join(root, "*.jpg")))
        self.patch_size = patch_size
        self.scale = scale
        self.transform = T.Compose([
            T.Resize(256),
            T.RandomCrop(patch_size),
            T.RandomHorizontalFlip(),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        hr = self.transform(img)  # (3, 224, 224)
        # 生成低分辨率图像 (scale=2 时是 112x112)
        lr = F.interpolate(
            hr.unsqueeze(0),
            scale_factor=1/self.scale,
            mode='bicubic',
            align_corners=False
        ).squeeze(0)
        return lr, hr

# =========================
# CLIP Wrapper（支持动态尺寸，用于提取 LR 特征）
# =========================
class CLIPViTFeature(nn.Module):
    def __init__(self, device):
        super().__init__()
        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained="openai"
        )
        self.visual = model.visual.to(device)
        self.visual.eval()
        for p in self.visual.parameters():
            p.requires_grad = False
        self.device = device
        self.embed_dim = 768
        self.patch_size = 16  # 关键参数

    @torch.no_grad()
    def forward(self, x):
        # x: (B, 3, H, W) 任意尺寸，H 和 W 必须能被 patch_size 整除
        B = x.shape[0]
        x = self.visual.conv1(x)  # (B, 768, H/16, W/16)
        H, W = x.shape[-2:]
        x = x.reshape(B, self.embed_dim, -1).permute(0, 2, 1)  # (B, N, 768)
        cls = self.visual.class_embedding.to(x.dtype).unsqueeze(0).expand(B, 1, -1)
        x = torch.cat([cls, x], dim=1)
        # 动态位置编码插值
        N = x.size(1)
        pos_embed = self.visual.positional_embedding.to(x.dtype)
        if N != 197:
            # 原位置编码是 (197, 768)，去掉 CLS 后为 (196, 768) -> (14,14,768)
            pos_patch = pos_embed[1:, :].reshape(1, 14, 14, 768).permute(0, 3, 1, 2)
            pos_patch = F.interpolate(pos_patch, size=(H, W), mode='bilinear')
            pos_patch = pos_patch.permute(0, 2, 3, 1).reshape(-1, 768)
            cls_pos = pos_embed[0:1, :]
            pos_embed = torch.cat([cls_pos, pos_patch], dim=0)
        x = x + pos_embed.unsqueeze(0)
        x = self.visual.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.visual.transformer(x)
        x = x.permute(1, 0, 2)
        patch = x[:, 1:, :]  # (B, N, 768)
        feat = patch.reshape(B, H, W, self.embed_dim).permute(0, 3, 1, 2).contiguous()
        return feat

# =========================
# 训练主函数（严格对齐原始 JAFAR）
# =========================
def train(data_root, epochs=50, batch_size=4, lr=2e-4, scale=2):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = ImageFolderDataset(data_root, patch_size=224, scale=scale)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    model = SatUp(dim=128, v_dim=768).to(device)
    clip_feat = CLIPViTFeature(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = Cosine_MSE().to(device)

    model.train()

    for epoch in range(epochs):
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        for i, (lr_img, hr_img) in enumerate(pbar):
            lr_img = lr_img.to(device)   # (B, 3, 112, 112) 当 scale=2
            hr_img = hr_img.to(device)   # (B, 3, 224, 224)

            # ----- 1. 提取高分辨率特征 (目标) -----
            with torch.no_grad():
                hr_feat = clip_feat(hr_img)  # (B, 768, 14, 14)

            # ----- 2. 模拟原始 backbone_feats：随机下采样 lr_img 提取低分辨率特征 -----
            # 原始代码对 224x224 图像随机下采样 0.25~0.5 倍；
            # 现在 lr_img 是 112x112，因此随机因子设为 0.4~0.8 倍，以确保 lr_feat 远小于 14x14
            down_factor = np.random.uniform(0.4, 0.8)
            _, _, h_lr, w_lr = lr_img.shape
            # 确保新尺寸是 patch_size (16) 的倍数，避免 reshape 报错
            new_h = (int(h_lr * down_factor) // 16) * 16
            new_w = (int(w_lr * down_factor) // 16) * 16
            # 防止尺寸为 0
            if new_h < 16 or new_w < 16:
                new_h, new_w = 16, 16

            low_res_img = F.interpolate(
                lr_img, size=(new_h, new_w), mode='bilinear', align_corners=False
            )
            with torch.no_grad():
                lr_feat = clip_feat(low_res_img)  # (B, 768, new_h/16, new_w/16)

            pred = model(
                image=low_res_img,
                features=lr_feat,
                output_size=hr_feat.shape[-2:]
            )

            # ----- 4. 损失计算（直接比较，形状完全一致）-----
            loss = criterion(pred, hr_feat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

            if i % 10 == 0:
                print(f"[E{epoch} | {i}] loss={loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch}: avg loss = {avg_loss:.4f}")
        torch.save(model.state_dict(), f"satup_vitb16_{epoch}.pth")

# =========================
# main
# =========================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--scale", type=int, default=2)
    args = parser.parse_args()

    train(
        data_root=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        scale=args.scale,
    )