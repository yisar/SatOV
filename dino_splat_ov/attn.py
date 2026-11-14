from pathlib import Path
import sys
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from einops import rearrange
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from torchvision.transforms import Normalize, ToTensor
from scipy.ndimage import gaussian_filter
import argparse
import os
import glob

root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))
from dino_splat_ov.gmm import gmm_fitting
from dino_splat_ov.dinov3.hub.dinotxt import dinov3_vitl16_dinotxt_tet1280d20h24l
from dino_splat_ov.gsup import GaussianFeatureUpsampler, create_coordinate_grid_2d

model, tokenizer = dinov3_vitl16_dinotxt_tet1280d20h24l()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device).eval()

win_size = 224
stride = 112

class_groups = [
    ["bus,car"],
    ["road"],
    ["bareland", "barren"],
    ["river", "water", "pool"],
    ["grass"],
    ["forest", "vegetation", "tree"],
    ["field", "cropland"],
    ["building", "roof"],
]

PURPLE_YELLOW = LinearSegmentedColormap.from_list(
    "purple_yellow",
    [
        (0.22, 0.05, 0.42), (0.42, 0.10, 0.58),
        (0.65, 0.20, 0.55), (0.85, 0.40, 0.35), (1.00, 0.65, 0.15),
        (1.00, 0.88, 0.10), (1.00, 1.00, 0.70),
    ],
)

# ---------------- 文本特征 ----------------
flat_texts = []
group_index_maps = []
for group in class_groups:
    start = len(flat_texts)
    flat_texts.extend(group)
    group_index_maps.append(list(range(start, len(flat_texts))))

texts = [f"a photo of {c}" for c in flat_texts]
num_classes = len(class_groups)

with torch.no_grad():
    text_tokens_all = tokenizer.tokenize(texts).to(device)
    text_feat_all = F.normalize(model.encode_text(text_tokens_all)[:, 1024:], dim=-1)

group_text_feats = []
for idx_list in group_index_maps:
    group_feat = text_feat_all[idx_list].mean(dim=0, keepdim=True)
    group_text_feats.append(group_feat)
group_text_feats = torch.cat(group_text_feats, dim=0)


def resolve_target_index(name):
    name_l = name.lower().strip()
    for gi, group in enumerate(class_groups):
        for c in group:
            if c.lower() == name_l:
                return gi
    for gi, group in enumerate(class_groups):
        for c in group:
            if name_l in c.lower() or c.lower() in name_l:
                return gi
    avail = " | ".join(" / ".join(g) for g in class_groups)
    raise ValueError(f"Class '{name}' not found.\nAvailable: {avail}")


def pad_to_multiple(img, win_size, stride):
    h, w = img.shape[:2]
    pad_h = (win_size - h) % stride
    pad_w = (win_size - w) % stride
    pad_top, pad_left = pad_h // 2, pad_w // 2
    img_padded = np.pad(
        img,
        ((pad_top, pad_h - pad_top), (pad_left, pad_w - pad_left), (0, 0)),
        mode="reflect",
    )
    return img_padded, pad_top, pad_left


def preprocess_patch(patch_np):
    normalize = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    tensor = ToTensor()(Image.fromarray(patch_np))
    return normalize(tensor).unsqueeze(0).to(device)


def _normalize_map(prob, norm_mode, p_lo, p_hi):
    if norm_mode == "minmax":
        tmin, tmax = float(prob.min()), float(prob.max())
        if tmax - tmin > 1e-8:
            return (prob - tmin) / (tmax - tmin)
        return np.zeros_like(prob)
    lo, hi = np.percentile(prob, p_lo), np.percentile(prob, p_hi)
    if hi - lo > 1e-8:
        return np.clip((prob - lo) / (hi - lo), 0.0, 1.0)
    return np.zeros_like(prob)


# ============================================================
# 单图推理 + 四分支可视化（无 TLP，GSUP 带窗口内 TTO fit）
# ============================================================
def predict_attention(image_path, target_idx, out_base, show=False,
                      sigma=2.0, norm_mode="minmax", p_lo=2.0, p_hi=98.0,
                      gmm_max_iter=5, gmm_temp=50.0,
                      gsup_fit_steps=5, gsup_fit_lr=1e2,
                      save_mask=False, mask_thresh=0.5):
    """
    输出：
      {out_base}_input.png   原图
      {out_base}_raw.png     原热力图（文本 logits + bilinear，无 GMM/GSUP）
      {out_base}_gmm.png     GMM 热力图（窗口内 GMM + bilinear）
      {out_base}_gsup.png    GSUP 热力图（窗口内 GMM + GSUP，含 TTO fit）
      {out_base}_compare.png 四联对比
    """
    image = Image.open(image_path).convert("RGB")
    W_img, H_img = image.size
    img_np = np.array(image)

    img_padded, pad_top, pad_left = pad_to_multiple(img_np, win_size, stride)
    H_pad, W_pad = img_padded.shape[:2]

    windows = []
    for y in range(0, H_pad - win_size + 1, stride):
        for x in range(0, W_pad - win_size + 1, stride):
            windows.append((img_padded[y:y + win_size, x:x + win_size, :],
                            y - pad_top, x - pad_left))

    # 三分支累积器
    up_raw_list, w_raw_list = [], []
    up_gmm_list, w_gmm_list = [], []
    up_gsup_list = []
    positions = []

    with torch.no_grad():
        for win_np, y0, x0 in windows:
            if y0 >= H_img or x0 >= W_img or y0 + win_size <= 0 or x0 + win_size <= 0:
                continue

            win_tensor = preprocess_patch(win_np)
            _, image_patch_tokens, _ = model.encode_image_with_patch_tokens(win_tensor)
            B, P, D = image_patch_tokens.shape
            h = w = int(P ** 0.5)

            img_feat = image_patch_tokens.transpose(1, 2).reshape(B, D, h, w)
            text_feat = F.normalize(text_feat_all, dim=-1)
            img_feat_n = F.normalize(img_feat, dim=1)
            logits_flat = torch.einsum("bchw,nc->bnhw", img_feat_n, text_feat)

            # -------- 分组 logits（原始语义） --------
            merged = []
            for idx_list in group_index_maps:
                wts = [1.0] + [0.3] * (len(idx_list) - 1)
                s = torch.zeros_like(logits_flat[:, idx_list[0], :, :])
                for i, idx in enumerate(idx_list):
                    s = s + logits_flat[:, idx, :, :] * wts[i]
                merged.append((s / sum(wts)).unsqueeze(1))
            logits_raw = torch.cat(merged, dim=1)            # [1,C,h,w]

            # -------- 窗口内 GMM --------
            dino_feats_flat = image_patch_tokens.reshape(-1, D).float()

            y_hat_flat = logits_raw.permute(0, 2, 3, 1).reshape(-1, num_classes).float()
            z = gmm_fitting(dino_feats_flat, y_hat_flat, temp=gmm_temp, max_iter=gmm_max_iter)
            z_map = z.reshape(1, h, w, num_classes).permute(0, 3, 1, 2)   # [1,C,h,w]
            log_z = torch.log(z_map.clamp(min=1e-6))          # 转 log 空间做插值

            # -------- 三个分支上采样 --------
            # 分支 1：raw bilinear
            up_raw = F.interpolate(logits_raw, size=(win_size, win_size),
                                   mode='bilinear', align_corners=False).squeeze(0)
            # 分支 2：GMM bilinear
            up_gmm = F.interpolate(log_z, size=(win_size, win_size),
                                   mode='bilinear', align_corners=False).squeeze(0)

            # 分支 3：GMM + GSUP（带窗口内 TTO fit）
            logits_lr = rearrange(log_z[0], "c h w -> (h w) c").unsqueeze(0)
            coords_lr = create_coordinate_grid_2d(h, w, device).reshape(-1, 2).unsqueeze(0)
            coords_hr = create_coordinate_grid_2d(win_size, win_size, device).reshape(-1, 2).unsqueeze(0)
            image_lr = F.interpolate(win_tensor, size=(h, w),
                                     mode="bilinear", align_corners=False)
            pixels_lr = rearrange(image_lr, "b c h w -> b (h w) c")
            pixels_hr = rearrange(win_tensor, "b c h w -> b (h w) c")

            # 1) 开梯度构建 + fit（外层 no_grad 被 enable_grad 覆盖）
            with torch.enable_grad():
                upsampler = GaussianFeatureUpsampler(
                    patch_coords_lr=coords_lr, patch_coords_hr=coords_hr,
                    pixels_lr=pixels_lr, pixels_hr=pixels_hr,
                ).to(device)
                if gsup_fit_steps > 0:
                    upsampler.fit(steps=gsup_fit_steps, lr=gsup_fit_lr, callback=None)

            # 2) 回到 no_grad，用优化后的参数做最终上采样
            with torch.no_grad():
                up_gsup = upsampler.forward(logits_lr) \
                                   .reshape(1, win_size, win_size, num_classes) \
                                   .permute(0, 3, 1, 2).squeeze(0)

            # -------- 各分支置信度权重 --------
            # raw 分支：用原始 logits 的 max softmax 概率
            w_raw = F.interpolate(
                logits_raw.softmax(dim=1).max(dim=1, keepdim=True)[0],
                size=(win_size, win_size), mode='bilinear'
            ).squeeze(0).squeeze(0)
            # GMM / GSUP 分支：用 GMM 输出的 max 概率
            w_gmm = F.interpolate(
                z_map.max(dim=1, keepdim=True)[0],
                size=(win_size, win_size), mode='bilinear'
            ).squeeze(0).squeeze(0)

            up_raw_list.append(up_raw.cpu())
            up_gmm_list.append(up_gmm.cpu())
            up_gsup_list.append(up_gsup.cpu())
            w_raw_list.append(w_raw.cpu())
            w_gmm_list.append(w_gmm.cpu())
            positions.append((y0, x0))

    if not positions:
        raise RuntimeError("No valid window processed.")

    # ============ 通用窗口融合 ============
    hann_1d = torch.hann_window(win_size)
    hann_2d = torch.outer(hann_1d, hann_1d)
    temperature = 0.2

    def fuse(logits_list, weight_list):
        acc_logits = torch.zeros((num_classes, H_img, W_img), dtype=torch.float32)
        acc_w = torch.zeros((H_img, W_img), dtype=torch.float32)
        for i, (y0, x0) in enumerate(positions):
            exp_w = torch.exp(weight_list[i] / temperature) * hann_2d
            ys, ye = max(0, y0), min(H_img, y0 + win_size)
            xs, xe = max(0, x0), min(W_img, x0 + win_size)
            cy1, cy2 = ys - y0, ys - y0 + (ye - ys)
            cx1, cx2 = xs - x0, xs - x0 + (xe - xs)
            acc_logits[:, ys:ye, xs:xe] += logits_list[i][:, cy1:cy2, cx1:cx2] * \
                                           exp_w[cy1:cy2, cx1:cx2].unsqueeze(0)
            acc_w[ys:ye, xs:xe] += exp_w[cy1:cy2, cx1:cx2]
        return acc_logits / acc_w.clamp(min=1e-6)

    final_raw  = fuse(up_raw_list,  w_raw_list)
    final_gmm  = fuse(up_gmm_list,  w_gmm_list)
    final_gsup = fuse(up_gsup_list, w_gmm_list)

    # ============ target 通道 → 归一化 → 平滑 ============
    def to_heat(final_logits):
        prob = F.softmax(final_logits, dim=0)[target_idx].cpu().numpy()
        heat = _normalize_map(prob, norm_mode, p_lo, p_hi)
        if sigma > 0:
            heat = gaussian_filter(heat, sigma=sigma)
        return heat

    heat_raw  = to_heat(final_raw)
    heat_gmm  = to_heat(final_gmm)
    heat_gsup = to_heat(final_gsup)

    # ============ 保存 4 张图 + 对比图 ============
    Image.fromarray(img_np).save(f"{out_base}_input.png")
    Image.fromarray((PURPLE_YELLOW(heat_raw)[..., :3] * 255).astype(np.uint8)).save(f"{out_base}_raw.png")
    Image.fromarray((PURPLE_YELLOW(heat_gmm)[..., :3] * 255).astype(np.uint8)).save(f"{out_base}_gmm.png")
    Image.fromarray((PURPLE_YELLOW(heat_gsup)[..., :3] * 255).astype(np.uint8)).save(f"{out_base}_gsup.png")

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    axes[0].imshow(image); axes[0].set_title(f"Input\n{class_groups[target_idx]}", fontsize=13); axes[0].axis("off")
    axes[1].imshow(heat_raw, cmap=PURPLE_YELLOW, vmin=0, vmax=1)
    axes[1].set_title("Raw text logits\n(bilinear)", fontsize=13); axes[1].axis("off")
    axes[2].imshow(heat_gmm, cmap=PURPLE_YELLOW, vmin=0, vmax=1)
    axes[2].set_title("GMM\n(window, bilinear)", fontsize=13); axes[2].axis("off")
    axes[3].imshow(heat_gsup, cmap=PURPLE_YELLOW, vmin=0, vmax=1)
    axes[3].set_title("GMM + GSUP\n(final, TTO)", fontsize=13); axes[3].axis("off")
    plt.tight_layout()
    plt.savefig(f"{out_base}_compare.png", bbox_inches='tight', dpi=150)
    if show:
        plt.show()
    plt.close()

    # ============ 可选 mask ============
    if save_mask:
        mask_bin = (heat_gsup > mask_thresh).astype(np.uint8) * 255
        Image.fromarray(mask_bin).save(f"{out_base}_mask.png")
        overlay = img_np.copy()
        color = np.array([255, 235, 59], dtype=np.uint8)
        m = mask_bin > 0
        overlay[m] = (overlay[m] * 0.5 + color * 0.5).astype(np.uint8)
        Image.fromarray(overlay).save(f"{out_base}_mask_overlay.png")

    return {"raw": heat_raw, "gmm": heat_gmm, "gsup": heat_gsup}


def main():
    parser = argparse.ArgumentParser(
        description="DINOv3 attention with four-panel output (Input / Raw / GMM / GSUP). TLP removed. GSUP TTO always on."
    )
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", "-o")
    parser.add_argument("--class", "-c", dest="target_class", default="building")
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--norm", choices=["minmax", "percentile"], default="minmax")
    parser.add_argument("--p-lo", type=float, default=2.0)
    parser.add_argument("--p-hi", type=float, default=98.0)

    # GMM 超参
    parser.add_argument("--gmm-max-iter", type=int, default=5)
    parser.add_argument("--gmm-temp", type=float, default=50.0)

    parser.add_argument("--save-mask", action="store_true")
    parser.add_argument("--mask-thresh", type=float, default=0.5)
    args = parser.parse_args()

    target_idx = resolve_target_index(args.target_class)
    print(f"Target: '{args.target_class}' -> class_groups[{target_idx}]")
    print(f"TLP=REMOVED | GMM=ON | GSUP=ON (TTO, steps=5, lr=1e2) | norm={args.norm}")

    def _run(in_file, out_base, show):
        predict_attention(
            in_file, target_idx, out_base, show=show,
            sigma=args.sigma, norm_mode=args.norm,
            p_lo=args.p_lo, p_hi=args.p_hi,
            gmm_max_iter=args.gmm_max_iter, gmm_temp=args.gmm_temp,
            save_mask=args.save_mask, mask_thresh=args.mask_thresh,
        )

    if os.path.isfile(args.input):
        if args.output is None:
            d = os.path.dirname(args.input) or "."
            name, _ = os.path.splitext(os.path.basename(args.input))
            out_base = os.path.join(d, name)
        elif os.path.isdir(args.output):
            name, _ = os.path.splitext(os.path.basename(args.input))
            out_base = os.path.join(args.output, name)
        else:
            out_base, _ = os.path.splitext(args.output)
        _run(args.input, out_base, show=True)
        print(f"Saved: {out_base}_input.png / _raw.png / _gmm.png / _gsup.png / _compare.png")

    elif os.path.isdir(args.input):
        out_dir = args.output or args.input
        os.makedirs(out_dir, exist_ok=True)
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif")
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(args.input, ext)))
            files.extend(glob.glob(os.path.join(args.input, ext.upper())))
        # 排除本脚本产出的文件
        skip_suffixes = ("_input.png", "_raw.png", "_gmm.png", "_gsup.png",
                         "_compare.png", "_mask.png", "_mask_overlay.png", "_attn.png")
        files = [f for f in files if not any(
            os.path.basename(f).endswith(s) for s in skip_suffixes
        )]
        if not files:
            print(f"No images in {args.input}"); return
        for f in files:
            name, _ = os.path.splitext(os.path.basename(f))
            out_base = os.path.join(out_dir, name)
            _run(f, out_base, show=False)
            print(f"Processed {f} -> {out_base}_compare.png")
    else:
        print(f"Input path {args.input} does not exist.")


if __name__ == "__main__":
    main()