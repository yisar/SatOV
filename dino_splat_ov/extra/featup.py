import torch
from PIL import Image
import matplotlib.pyplot as plt


# =========================================================
# 1. Device
# =========================================================
# import os
# os.environ["GITHUB_TOKEN"] = "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"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")


# =========================================================
# 2. Load UPLiFT + DINOv2-S/14
# =========================================================

print("Loading UPLiFT model...")

model = torch.hub.load('mwalmer-umd/UPLiFT', 'uplift_dinov2_s14')


model = model.to(device)
model.eval()

print("Model loaded successfully.")


# =========================================================
# 3. Load image
# =========================================================

image_path = "asset/9.png"

image = Image.open(image_path).convert("RGB")

print(f"Input image size: {image.size}")


# =========================================================
# 4. Inference
# =========================================================

with torch.no_grad():

    features = model(image)


# =========================================================
# 5. Print output
# =========================================================

print("Output type:", type(features))

if isinstance(features, torch.Tensor):

    print("Feature shape:", features.shape)

else:

    print(features)


# =========================================================
# 6. Simple feature visualization
# =========================================================
# =========================================================
# 6. PCA Feature Visualization
# =========================================================

import numpy as np
from sklearn.decomposition import PCA


if isinstance(features, torch.Tensor):

    # -----------------------------------------------------
    # 处理输出维度
    # -----------------------------------------------------
    #
    # [B, C, H, W]
    # 或 [C, H, W]
    #

    if features.ndim == 3:
        features = features.unsqueeze(0)

    feature = features[0]  # [C, H, W]

    print("Feature shape:", feature.shape)

    # -----------------------------------------------------
    # [C, H, W] -> [H*W, C]
    # -----------------------------------------------------

    feature = feature.permute(1, 2, 0)

    H, W, C = feature.shape

    feature_flat = feature.reshape(-1, C)

    # -----------------------------------------------------
    # PCA: C维 -> 3维
    # -----------------------------------------------------

    pca = PCA(n_components=3)

    feature_pca = pca.fit_transform(
        feature_flat.detach().cpu().numpy()
    )

    print(
        "Explained variance ratio:",
        pca.explained_variance_ratio_
    )

    # -----------------------------------------------------
    # 每个 PCA 分量独立归一化
    # -----------------------------------------------------

    feature_pca = feature_pca.reshape(H, W, 3)

    feature_vis = np.zeros_like(feature_pca)

    for i in range(3):

        channel = feature_pca[..., i]

        channel_min = channel.min()
        channel_max = channel.max()

        feature_vis[..., i] = (
            channel - channel_min
        ) / (
            channel_max - channel_min + 1e-8
        )

    # -----------------------------------------------------
    # 可视化
    # -----------------------------------------------------

    plt.figure(figsize=(8, 8))

    plt.imshow(feature_vis)

    plt.title(
        "UPLiFT + DINOv2-S/14 PCA Feature"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.show()