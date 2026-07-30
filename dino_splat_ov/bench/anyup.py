import os
import torch
import matplotlib.pyplot as plt
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

    # 3. 加载AnyUp上采样模型
    upsampler = torch.hub.load("wimmerth/anyup", "anyup", verbose=False).to(device).eval()
    with torch.no_grad():
        hr_features = upsampler(hr_image, lr_features, q_chunk_size=256)

    # 4. 联合PCA降维可视化
    with torch.no_grad():
        # 展平特征
        lr_flat = lr_features[0].permute(1, 2, 0).reshape(-1, C)
        hr_flat = hr_features[0].permute(1, 2, 0).reshape(-1, C)
        all_feats = torch.cat([lr_flat, hr_flat], dim=0)

        # SVD实现PCA
        feat_mean = all_feats.mean(dim=0, keepdim=True)
        X = all_feats - feat_mean
        _, _, Vh = torch.linalg.svd(X, full_matrices=False)
        pcs = Vh[:3].T

        proj_all = X @ pcs
        n_lr_pix = h * w
        proj_lr = proj_all[:n_lr_pix].reshape(h, w, 3)
        proj_hr = proj_all[n_lr_pix:].reshape(img_size, img_size, 3)

        # 全局统一归一化
        cmin = proj_all.min(dim=0).values
        cmax = proj_all.max(dim=0).values
        crange = (cmax - cmin).clamp(min=1e-6)

        lr_rgb = ((proj_lr - cmin) / crange).cpu().numpy()
        hr_rgb = ((proj_hr - cmin) / crange).cpu().numpy()

    # 5. 绘图展示
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(lr_rgb)
    axs[0].set_title(f"LR DINOv2 Feature ({h}×{w})")
    axs[0].axis("off")

    axs[1].imshow(hr_rgb)
    axs[1].set_title(f"AnyUp Upsampled Feature ({img_size}×{img_size})")
    axs[1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()