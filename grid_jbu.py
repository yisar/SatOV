import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time


# =========================================================
# main API (保持你的接口不变)
# =========================================================
def GridJBU(HR_img, lr_modality):
    """
    FeatUp-style Bilateral Attention Upsampling (方案A增强版)
    input:
        HR_img: RGB image (numpy / PIL / torch)
        lr_modality: low-res feature map (B,C,h,w)
    return:
        hr_feat: (B,C,H,W)
    """

    start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # 1. preprocess HR image
    # -------------------------
    if isinstance(HR_img, torch.Tensor):
        hr = HR_img.detach().float().cpu()
        if hr.dim() == 3:
            hr = hr.unsqueeze(0)
    else:
        hr = torch.from_numpy(np.array(HR_img)).float()
        if hr.ndim == 3:
            hr = hr.permute(2, 0, 1).unsqueeze(0)

    hr = hr.to(device) / 255.0
    B, _, H, W = hr.shape

    # -------------------------
    # 2. preprocess LR feature
    # -------------------------
    if isinstance(lr_modality, torch.Tensor):
        feat_lr = lr_modality.to(device).float()
    else:
        feat_lr = torch.from_numpy(np.array(lr_modality)).float().to(device)

    if feat_lr.dim() == 3:
        feat_lr = feat_lr.unsqueeze(0)

    B, C, Hl, Wl = feat_lr.shape

    scale = H // Hl

    # =========================================================
    # 3. Feature upsample baseline (very important residual)
    # =========================================================
    feat_up = F.interpolate(
        feat_lr,
        size=(H, W),
        mode="bilinear",
        align_corners=False
    )

    # =========================================================
    # 4. RGB guidance (important improvement over your code)
    # =========================================================
    guide = F.interpolate(hr, size=(H, W), mode="bilinear", align_corners=False)

    # normalize guide
    guide_mean = guide.mean(dim=(2, 3), keepdim=True)
    guide = guide - guide_mean

    # =========================================================
    # 5. local bilateral attention window
    # =========================================================
    radius = 3              # 7x7 window
    sigma_spatial = 2.0
    sigma_range = 0.08

    # unfold feature (low-res grid aligned to HR)
    feat_lr_up = F.interpolate(feat_lr, size=(H, W), mode="nearest")

    # pad
    pad = radius
    feat_pad = F.pad(feat_lr_up, (pad, pad, pad, pad))
    guide_pad = F.pad(guide, (pad, pad, pad, pad))

    out = torch.zeros_like(feat_lr_up)

    norm = torch.zeros((B, 1, H, W), device=device)

    # =========================================================
    # 6. bilateral propagation (核心改进点)
    # =========================================================
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):

            shifted_feat = feat_pad[:, :, pad + dy: pad + dy + H, pad + dx: pad + dx + W]
            shifted_guide = guide_pad[:, :, pad + dy: pad + dy + H, pad + dx: pad + dx + W]

            # spatial weight
            spatial_w = math.exp(-(dx * dx + dy * dy) / (2 * sigma_spatial ** 2))

            # range weight (RGB guidance)
            range_dist = (guide - shifted_guide).pow(2).sum(dim=1, keepdim=True)
            range_w = torch.exp(-range_dist / (2 * sigma_range ** 2))

            weight = spatial_w * range_w

            out = out + shifted_feat * weight
            norm = norm + weight

    out = out / (norm + 1e-6)

    # =========================================================
    # 7. residual refinement (FeatUp核心思想)
    # =========================================================
    hr_feat = out + feat_up

    # light sharpening (optional but useful)
    blur = F.avg_pool2d(hr_feat, kernel_size=3, stride=1, padding=1)
    hr_feat = hr_feat + 0.2 * (hr_feat - blur)

    print(f"[GridJBU-FeatUp] done in {time.time() - start_time:.2f}s")

    return hr_feat