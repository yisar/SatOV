import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import open_clip
from model import SatUp   # 请确保 model.py 中定义了 SatUp 类


# =========================
# Stable PCA (GT anchor)
# =========================
class FixedPCA:
    def __init__(self, n_components=3):
        self.n_components = n_components

    def fit(self, X):
        self.mean_ = X.mean(dim=0, keepdim=True)
        Xc = X - self.mean_

        U, S, V = torch.pca_lowrank(Xc, q=self.n_components, center=False)
        self.components_ = V[:, :self.n_components]

        # sign fix for stability
        for i in range(self.components_.shape[1]):
            if self.components_[:, i].sum() < 0:
                self.components_[:, i] *= -1

        return self

    def transform(self, X):
        return (X - self.mean_) @ self.components_


def flatten_feat(feat):
    B, C, H, W = feat.shape
    return feat.permute(0, 2, 3, 1).reshape(-1, C).detach().cpu()


def stable_pca(gt_feat, pred_feat, lr_feat):
    """基于 gt_feat 拟合 PCA，并变换 pred 和 lr"""
    X = flatten_feat(gt_feat)
    pca = FixedPCA(3).fit(X)

    def apply(feat):
        x = flatten_feat(feat)
        x = pca.transform(x)

        x = x - x.min(dim=0, keepdim=True).values
        x = x / (x.max(dim=0, keepdim=True).values + 1e-8)

        B, C, H, W = feat.shape
        return x.reshape(B, H, W, 3).permute(0, 3, 1, 2)

    return apply(gt_feat), apply(pred_feat), apply(lr_feat)


def pca_transform_single(feat, n_components=3):
    """对单一特征图进行独立 PCA（不依赖其他特征）"""
    X = flatten_feat(feat)
    pca = FixedPCA(n_components).fit(X)
    x = pca.transform(X)
    x = x - x.min(dim=0, keepdim=True).values
    x = x / (x.max(dim=0, keepdim=True).values + 1e-8)
    B, C, H, W = feat.shape
    return x.reshape(B, H, W, n_components).permute(0, 3, 1, 2)


# =========================
# ClearCLIP Feature Extractor
# =========================
class ClearCLIPFeature(nn.Module):
    def __init__(self, model_name="ViT-B-16", device="cuda"):
        super().__init__()
        self.device = device

        model, _, _ = open_clip.create_model_and_transforms(
            model_name,
            pretrained="openai",
            device=device
        )

        self.visual = model.visual.eval()

        for p in self.visual.parameters():
            p.requires_grad = False

        self.embed_dim = self.visual.conv1.out_channels

    @torch.no_grad()
    def forward(self, x):
        B, C, H, W = x.shape

        x = self.visual.conv1(x)
        grid_h, grid_w = x.shape[-2:]

        x_tokens = x.flatten(2).transpose(1, 2)

        cls_token = self.visual.class_embedding.to(x_tokens.dtype)
        pos_embed = self.visual.positional_embedding.to(x_tokens.dtype)

        cls_pos = pos_embed[:1]
        patch_pos = pos_embed[1:]

        old_grid = int(patch_pos.shape[0] ** 0.5)

        if old_grid != grid_h or old_grid != grid_w:
            patch_pos = patch_pos.reshape(1, old_grid, old_grid, -1).permute(0, 3, 1, 2)
            patch_pos = F.interpolate(
                patch_pos, size=(grid_h, grid_w),
                mode="bicubic",
                align_corners=False
            )
            patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, grid_h * grid_w, -1)
            pos_embed = torch.cat([cls_pos, patch_pos.squeeze(0)], dim=0)

        x_tokens = torch.cat(
            [cls_token.unsqueeze(0).expand(B, 1, -1), x_tokens],
            dim=1
        )

        x_tokens = x_tokens + pos_embed
        x_tokens = self.visual.ln_pre(x_tokens)

        blocks = self.visual.transformer.resblocks
        for i in range(len(blocks) - 1):
            x_tokens = blocks[i](x_tokens)

        last = blocks[-1]

        x = last.ln_1(x_tokens)
        attn = last.attn

        qkv = F.linear(x, attn.in_proj_weight, attn.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)

        B, N, D = q.shape
        heads = attn.num_heads
        dim_head = D // heads

        q = q.view(B, N, heads, dim_head).transpose(1, 2)
        v = v.view(B, N, heads, dim_head).transpose(1, 2)

        attn_map = (q @ q.transpose(-2, -1)) * (dim_head ** -0.5)
        attn_map = attn_map.softmax(dim=-1)

        out = (attn_map @ v).transpose(1, 2).reshape(B, N, D)

        out = F.linear(out, attn.out_proj.weight, attn.out_proj.bias)
        out = self.visual.ln_post(out)

        cls = out[:, :1]
        patch = out[:, 1:]
        patch = patch - cls

        feat = patch.permute(0, 2, 1).reshape(B, D, grid_h, grid_w)

        return feat


# =========================
# main
# =========================
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    # ---------- 加载 ClearCLIP ----------
    clip_encoder = ClearCLIPFeature(
        model_name="ViT-B-16",
        device=device
    ).to(device)
    clip_encoder.eval()

    # ---------- 加载 DINOv2 ----------
    print("Loading DINOv2 model...")
    dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', pretrained=True).to(device)
    dino.eval()
    dino_norm = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])

    # ---------- 读取并预处理图像 ----------
    img = Image.open(args.image).convert("RGB")
    hr = T.ToTensor()(img).unsqueeze(0).to(device)
    hr = F.interpolate(hr, (args.output_size, args.output_size), mode="bilinear")

    lr = F.interpolate(hr, scale_factor=0.5, mode="bicubic")

    # ---------- ClearCLIP 特征 ----------
    gt_feat = clip_encoder(hr)
    lr_feat = clip_encoder(lr)

    # ---------- DINOv2 特征 (带位置编码插值，兼容任意分辨率) ----------
    hr_norm = dino_norm(hr)
    with torch.no_grad():
        x = dino.patch_embed(hr_norm)                 # (B, N, D)
        cls_token = dino.cls_token.expand(hr_norm.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)          # (B, N+1, D)

        # 动态调整位置编码尺寸
        pos_embed = dino.pos_embed                    # (1, old_N+1, D)
        old_N = pos_embed.shape[1] - 1
        old_grid = int(old_N ** 0.5)
        cur_N = x.shape[1] - 1
        cur_grid = int(cur_N ** 0.5)

        if old_grid != cur_grid:
            cls_pos = pos_embed[:, :1, :]
            patch_pos = pos_embed[:, 1:, :]           # (1, old_N, D)
            patch_pos = patch_pos.reshape(1, old_grid, old_grid, -1).permute(0, 3, 1, 2)  # (1, D, old_grid, old_grid)
            patch_pos = F.interpolate(patch_pos, size=(cur_grid, cur_grid), mode='bicubic', align_corners=False)
            patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, cur_grid*cur_grid, -1)
            pos_embed_new = torch.cat([cls_pos, patch_pos], dim=1)
        else:
            pos_embed_new = pos_embed

        x = x + pos_embed_new

        for blk in dino.blocks:
            x = blk(x)
        x = dino.norm(x)                              # (B, N+1, D)
        patch = x[:, 1:]                              # (B, N, D)
        N = patch.shape[1]
        grid_size = int(N ** 0.5)
        dino_feat = patch.permute(0, 2, 1).reshape(hr.shape[0], -1, grid_size, grid_size)

    # ---------- SatUp 预测 ----------
    satup = SatUp(dim=128, v_dim=768).to(device)
    satup.load_state_dict(torch.load(args.weight, map_location=device, weights_only=True))
    satup.eval()

    with torch.no_grad():
        pred_feat = satup(
            lr,
            lr_feat,
            (args.output_size, args.output_size)
        )

    # 上采样用于可视化（ClearCLIP 的 LR 特征上采样，仅用于稳定 PCA 的锚点）
    lr_feat_up = F.interpolate(lr_feat, (args.output_size, args.output_size), mode="nearest")
    gt_feat_up = F.interpolate(gt_feat, (args.output_size, args.output_size), mode="nearest")

    # ---------- PCA 可视化 ----------
    # 1) ClearCLIP 的 GT、Pred 和 LR（以 GT 为锚点）
    gt_pca, pred_pca, lr_pca = stable_pca(gt_feat_up, pred_feat, lr_feat_up)   # 第三返回值现为 LR 特征

    # 2) DINOv2 特征独立 PCA
    dino_pca = pca_transform_single(dino_feat, n_components=3)

    # 转为 numpy 图像
    img_np = hr[0].permute(1, 2, 0).detach().cpu().numpy()
    gt_np = gt_pca[0].permute(1, 2, 0).detach().cpu().numpy()
    pred_np = pred_pca[0].permute(1, 2, 0).detach().cpu().numpy()
    dino_np = dino_pca[0].permute(1, 2, 0).detach().cpu().numpy()
    lr_np = lr_pca[0].permute(1, 2, 0).detach().cpu().numpy()                  # 新增 LR PCA 结果

    # ---------- 绘图 ----------
    fig, ax = plt.subplots(1, 4, figsize=(18, 5))

    ax[0].imshow(img_np)
    ax[0].set_title("Image")

    ax[2].imshow(dino_np)
    ax[2].set_title("DINOv2 HR")

    ax[1].imshow(gt_np)                                                         # 显示 LR 特征图 PCA
    ax[1].set_title("CLIP HR")                                                # 修改标题

    ax[3].imshow(pred_np)
    ax[3].set_title("SatUp")

    for a in ax:
        a.axis("off")

    plt.tight_layout()
    save_path = Path(args.save_dir) / f"dinov2_pca_{args.output_size}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print("Saved to:", save_path)


# =========================
# run
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="asset/P0016654.jpg")
    parser.add_argument("--weight", default="satup_a_15.pth")
    parser.add_argument("--save_dir", default="results")
    parser.add_argument("--output_size", type=int, default=224)

    args = parser.parse_args()
    main(args)