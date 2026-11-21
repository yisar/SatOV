import os
import matplotlib.pyplot as plt
import numpy as np
import open_clip
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm  # 保留与训练脚本一致的依赖风格

from model import SatUp

# ===================== 配置区（按需修改）=====================
local_img_path = "asset/004.tif"  # 多波段 TIF 路径
rgb_band_indices = (0, 1, 2)      # 用于提取 ClearCLIP 特征的波段索引
img_size = 448                    # 输入统一分辨率
clip_model_name = "ViT-B-16"      # 与训练脚本一致的 open_clip 模型名
clip_pretrained = "openai"        # 与训练脚本一致的预训练权重
ckpt_path = "satup.pth"                  # 可选：训练好的 SatUp 权重路径，如 "satup_a_49.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
# ==========================================================


class ClearCLIPFeature(nn.Module):
    def __init__(self, model_name="ViT-B-16", pretrained="openai", device="cuda"):
        super().__init__()
        self.device = device

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=device,
        )
        self.preprocess = preprocess

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
                patch_pos,
                size=(grid_h, grid_w),
                mode="bicubic",
                align_corners=False,
            )
            patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, grid_h * grid_w, -1)
            pos_embed = torch.cat([cls_pos, patch_pos.squeeze(0)], dim=0)

        x_tokens = torch.cat(
            [cls_token.unsqueeze(0).expand(B, 1, -1), x_tokens],
            dim=1,
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


def load_multiband_tif(tif_path: str) -> np.ndarray:
    """读取多波段 TIF，返回 float32 数组，shape 为 [C, H, W]"""
    with rasterio.open(tif_path) as src:
        data = src.read().astype(np.float32)
    return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)


def bands_to_rgb_pil(band_chw: np.ndarray, rgb_indices: tuple[int, int, int]) -> Image.Image:
    """将多波段数组转为 uint8 RGB PIL Image"""
    c, _, _ = band_chw.shape
    idxs = [min(i, c - 1) for i in rgb_indices]
    rgb = np.stack([band_chw[i] for i in idxs], axis=-1)

    out = np.zeros_like(rgb, dtype=np.float32)
    for ch in range(3):
        channel = rgb[..., ch]
        p_low, p_high = np.percentile(channel, [2, 98])
        if p_high - p_low > 1e-6:
            out[..., ch] = np.clip((channel - p_low) / (p_high - p_low), 0.0, 1.0)
        else:
            c_min, c_max = channel.min(), channel.max()
            if c_max - c_min > 1e-6:
                out[..., ch] = np.clip((channel - c_min) / (c_max - c_min), 0.0, 1.0)

    return Image.fromarray((out * 255.0).astype(np.uint8), mode="RGB")


def prepare_clip_rgb_tensor(rgb_pil: Image.Image, mean, std, target_size: int, dev: str) -> torch.Tensor:
    """按训练脚本的 mean / std 归一化 RGB 图像"""
    resized = rgb_pil.resize((target_size, target_size), Image.BICUBIC)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(dev)
    mean_t = torch.tensor(mean, device=dev, dtype=torch.float32).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=dev, dtype=torch.float32).view(1, 3, 1, 1)
    return (tensor - mean_t) / std_t


def normalize_bands_for_upsampler(band_chw: np.ndarray, target_h: int, target_w: int, dev: str) -> torch.Tensor:
    """将多波段高分影像调整到 (target_h, target_w) 并做波段级标准化，返回 [1, C, H, W]"""
    tensor = torch.from_numpy(band_chw).unsqueeze(0).to(device=dev, dtype=torch.float32)
    tensor = F.interpolate(tensor, size=(target_h, target_w), mode="bilinear", align_corners=False)

    mean = tensor.mean(dim=(-2, -1), keepdim=True)
    std = tensor.std(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
    return (tensor - mean) / std


def pca_rgb(feat_chw: torch.Tensor, h_out: int, w_out: int) -> np.ndarray:
    c, _, _ = feat_chw.shape
    flat = feat_chw.permute(1, 2, 0).reshape(-1, c)
    mean = flat.mean(dim=0, keepdim=True)
    x_centered = flat - mean
    _, _, vh = torch.linalg.svd(x_centered, full_matrices=False)
    pcs = vh[:3].T
    proj = (x_centered @ pcs).reshape(h_out, w_out, 3)

    p_min = proj.min(dim=0).values.min(dim=0).values
    p_max = proj.max(dim=0).values.max(dim=0).values
    p_range = (p_max - p_min).clamp(min=1e-6)
    rgb = ((proj - p_min) / p_range).cpu().numpy()
    return np.clip(rgb * 1.1, 0, 1)


def main():
    if not os.path.exists(local_img_path):
        raise FileNotFoundError(f"图片文件不存在: {local_img_path}")

    multiband_chw = load_multiband_tif(local_img_path)
    num_bands = multiband_chw.shape[0]
    rgb_pil = bands_to_rgb_pil(multiband_chw, rgb_band_indices)

    # 1. 加载 ClearCLIPFeature 提取特征
    extractor = ClearCLIPFeature(
        model_name=clip_model_name,
        pretrained=clip_pretrained,
        device=device,
    )
    mean = extractor.preprocess.transforms[-1].mean
    std = extractor.preprocess.transforms[-1].std

    hr_rgb_image = prepare_clip_rgb_tensor(rgb_pil, mean, std, img_size, device)
    lr_features = extractor(hr_rgb_image)  # [B, C, h, w]
    _, C, h, w = lr_features.shape

    # 2. 准备任意波段高分辨率引导张量
    hr_multiband_guide = normalize_bands_for_upsampler(multiband_chw, img_size, img_size, device)

    # 3. 加载 SatUp 上采样模型
    upsampler = SatUp(dim=128, v_dim=C, kernel_size=3, num_heads=4).to(device).eval()
    if ckpt_path is not None and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        upsampler.load_state_dict(state, strict=False)

    with torch.no_grad():
        hr_features = upsampler(hr_multiband_guide, lr_features, output_size=(img_size, img_size))

    # 4. LR 与 HR 特征独立 PCA 可视化
    with torch.no_grad():
        lr_rgb = pca_rgb(lr_features[0], h, w)
        hr_rgb = pca_rgb(hr_features[0], img_size, img_size)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(lr_rgb)
    axs[0].set_title(f"LR ClearCLIP Feature ({h}×{w}, C={C})")
    axs[0].axis("off")

    axs[1].imshow(hr_rgb)
    axs[1].set_title(f"SatUp Upsampled ({num_bands} bands -> {img_size}×{img_size})")
    axs[1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()