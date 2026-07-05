import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, Dinov2Model

# ===================== 配置区（按需修改）=====================
local_img_path = "asset/9.png"  # 本地图片路径
img_size = 448                 # 输入统一分辨率
device = "cuda" if torch.cuda.is_available() else "cpu"
# ==========================================================


def main():
    # 1. 读取本地图片
    if not os.path.exists(local_img_path):
        raise FileNotFoundError(f"图片文件不存在: {local_img_path}")
    img = Image.open(local_img_path).convert("RGB")

    # 2. 加载DINOv2-base模型与预处理
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base", use_fast=True)
    dino_model = Dinov2Model.from_pretrained("facebook/dinov2-base").to(device).eval()

    # 图像预处理
    inputs = processor(
        images=img,
        do_resize=True,
        size={"shortest_edge": img_size},
        do_center_crop=True,
        crop_size={"height": img_size, "width": img_size},
        return_tensors="pt",
    )
    hr_image = inputs["pixel_values"].to(device)

    # 提取DINO特征token，去除CLS
    with torch.no_grad():
        outputs = dino_model(pixel_values=hr_image)
        tokens = outputs.last_hidden_state[:, 1:, :]

    B, N, C = tokens.shape
    h = w = int(N ** 0.5)
    assert h * w == N, f"Token数量{N}无法构成正方形网格"

    # 转换为卷积格式特征图 [B, C, H, W]
    lr_features = tokens.reshape(B, h, w, C).permute(0, 3, 1, 2).contiguous()

    # 3. 加载上采样模型（AnyUp示例，替换SATUp仅改hub加载行）
    upsampler = torch.hub.load("wimmerth/anyup", "anyup", verbose=False).to(device).eval()
    with torch.no_grad():
        hr_features = upsampler(hr_image, lr_features, q_chunk_size=256)

    # 4. LR、HR 独立分开PCA，不再联合，互不干扰对比度
    with torch.no_grad():
        # ---------------------- LR低分辨率特征单独PCA ----------------------
        lr_flat = lr_features[0].permute(1, 2, 0).reshape(-1, C)
        lr_mean = lr_flat.mean(dim=0, keepdim=True)
        X_lr = lr_flat - lr_mean
        _, _, Vh_lr = torch.linalg.svd(X_lr, full_matrices=False)
        pcs_lr = Vh_lr[:3].T
        proj_lr = (X_lr @ pcs_lr).reshape(h, w, 3)

        # 修复：分步求全局最小/最大，不传入tuple dim
        lr_min = proj_lr.min(dim=0).values.min(dim=0).values
        lr_max = proj_lr.max(dim=0).values.max(dim=0).values
        lr_range = (lr_max - lr_min).clamp(min=1e-6)
        lr_rgb = ((proj_lr - lr_min) / lr_range).cpu().numpy()
        lr_rgb = np.clip(lr_rgb * 1.1, 0, 1)

        # ---------------------- HR上采样特征单独PCA ----------------------
        hr_flat = hr_features[0].permute(1, 2, 0).reshape(-1, C)
        hr_mean = hr_flat.mean(dim=0, keepdim=True)
        X_hr = hr_flat - hr_mean
        _, _, Vh_hr = torch.linalg.svd(X_hr, full_matrices=False)
        pcs_hr = Vh_hr[:3].T
        proj_hr = (X_hr @ pcs_hr).reshape(img_size, img_size, 3)

        # 修复：分步求全局最小/最大
        hr_min = proj_hr.min(dim=0).values.min(dim=0).values
        hr_max = proj_hr.max(dim=0).values.max(dim=0).values
        hr_range = (hr_max - hr_min).clamp(min=1e-6)
        hr_rgb = ((proj_hr - hr_min) / hr_range).cpu().numpy()
        hr_rgb = np.clip(hr_rgb * 1.1, 0, 1)

    # 5. 绘图展示
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(lr_rgb)
    axs[0].set_title(f"LR DINOv2 Feature ({h}×{w})")
    axs[0].axis("off")

    axs[1].imshow(hr_rgb)
    axs[1].set_title(f"SATUp Upsampled Feature ({img_size}×{img_size})")
    axs[1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main() 