import sys
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
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
# ★ TLP 已移除
from dino_splat_ov.gmm import gmm_fitting
from dino_splat_ov.dinov3.hub.dinotxt import dinov3_vitl16_dinotxt_tet1280d20h24l
from dino_splat_ov.gsup import GaussianFeatureUpsampler, create_coordinate_grid_2d

model, tokenizer = dinov3_vitl16_dinotxt_tet1280d20h24l()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device).eval()

win_size = 224
stride = 112
PATCH_SIZE = 16

class_groups = [
    ["bus,car"], ["road"], ["bareland", "barren"],
    ["river", "water", "pool"], ["grass"],
    ["forest", "vegetation", "tree"], ["field", "cropland"],
    ["building", "roof"],
]

PURPLE_YELLOW = LinearSegmentedColormap.from_list(
    "purple_yellow",
    [
        (0.05, 0.00, 0.15), (0.22, 0.05, 0.42), (0.42, 0.10, 0.58),
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
    group_text_feats.append(text_feat_all[idx_list].mean(dim=0, keepdim=True))
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
# 核心：全局 GMM + 窗口内 GSUP（无 TLP）
# ============================================================
def predict_attention(image_path, target_idx, out_base, show=False,
                      sigma=2.0, norm_mode="minmax", p_lo=2.0, p_hi=98.0,
                      use_gsup=True, use_gmm=True,
                      gmm_max_iter=10, gmm_temp=50.0,
                      save_mask=False, mask_thresh=0.5):
    """
    三开关：use_gmm / use_gsup；TLP 已移除。
    输出：
      {out_base}_input.png     原图
      {out_base}_gmm.png       GMM 概率热力图（上采样到原图尺寸展示）
      {out_base}_gsup.png      GSUP 最终热力图
      {out_base}_compare.png   三联对比图
      [optional] _mask.png / _mask_overlay.png
    """
    image = Image.open(image_path).convert("RGB")
    W_img, H_img = image.size
    img_np = np.array(image)

    img_padded, pad_top, pad_left = pad_to_multiple(img_np, win_size, stride)
    H_pad, W_pad = img_padded.shape[:2]
    H_p = H_pad // PATCH_SIZE
    W_p = W_pad // PATCH_SIZE

    windows = []
    for y in range(0, H_pad - win_size + 1, stride):
        for x in range(0, W_pad - win_size + 1, stride):
            windows.append((y, x))

    # ============ 第一遍：特征累加 ============
    feat_acc = None
    yhat_acc = None
    count_acc = None
    window_cache = []

    with torch.no_grad():
        for y, x in windows:
            y0, x0 = y - pad_top, x - pad_left
            if y0 >= H_img or x0 >= W_img or y0 + win_size <= 0 or x0 + win_size <= 0:
                continue

            win_np = img_padded[y:y + win_size, x:x + win_size, :]
            win_tensor = preprocess_patch(win_np)
            _, image_patch_tokens, _ = model.encode_image_with_patch_tokens(win_tensor)
            B, P, D = image_patch_tokens.shape
            h = w = int(P ** 0.5)

            if feat_acc is None:
                feat_acc  = torch.zeros(D, H_p, W_p, dtype=torch.float32)
                yhat_acc  = torch.zeros(num_classes, H_p, W_p, dtype=torch.float32)
                count_acc = torch.zeros(1, H_p, W_p, dtype=torch.float32)

            img_feat = image_patch_tokens.transpose(1, 2).reshape(B, D, h, w)
            img_feat = F.normalize(img_feat, dim=1)

            text_feat = F.normalize(text_feat_all, dim=-1)
            logits_flat = torch.einsum("bchw,nc->bnhw", img_feat, text_feat)
            merged = []
            for idx_list in group_index_maps:
                weights = [1.0] + [0.3] * (len(idx_list) - 1)
                s = torch.zeros_like(logits_flat[:, idx_list[0], :, :])
                for i, idx in enumerate(idx_list):
                    s = s + logits_flat[:, idx, :, :] * weights[i]
                merged.append((s / sum(weights)).unsqueeze(1))
            logits = torch.cat(merged, dim=1)

            py = y // PATCH_SIZE
            px = x // PATCH_SIZE
            feat_acc[:, py:py + h, px:px + w]  += img_feat.squeeze(0).cpu()
            yhat_acc[:, py:py + h, px:px + w]  += logits.squeeze(0).cpu()
            count_acc[:, py:py + h, px:px + w] += 1.0

            window_cache.append({
                'y': y, 'x': x, 'y0': y0, 'x0': x0,
                'py': py, 'px': px, 'h': h, 'w': w,
            })

    if feat_acc is None:
        raise RuntimeError("No valid window processed.")

    feat_acc  /= count_acc.clamp(min=1)
    yhat_acc  /= count_acc.clamp(min=1)

    # ============ 全局 GMM ============
    if use_gmm:
        feats_flat = feat_acc.permute(1, 2, 0).reshape(-1, D).to(device).float()
        feats_flat = F.normalize(feats_flat, dim=-1)
        yhat_flat  = yhat_acc.permute(1, 2, 0).reshape(-1, num_classes).to(device).float()

        z = gmm_fitting(feats_flat, yhat_flat, temp=gmm_temp, max_iter=gmm_max_iter)
        z_map = z.reshape(H_p, W_p, num_classes).permute(2, 0, 1).unsqueeze(0)
        # ★ GMM 输出已是概率（z 每行 sum=1）
        gmm_probs = z_map
    else:
        z_map = yhat_acc.unsqueeze(0).to(device)
        # ★ 原始 logits 需要 softmax 转概率
        gmm_probs = F.softmax(z_map, dim=1)

    # ---- 提取 GMM 可视化用的低分辨率概率图 ----
    gmm_target_lr = gmm_probs[0, target_idx].cpu().numpy()           # [H_p, W_p]
    gmm_norm_lr = _normalize_map(gmm_target_lr, norm_mode, p_lo, p_hi)
    # 轻微平滑（GMM 本身在 patch 级别，已经较平滑，不需要强 sigma）
    if sigma > 0:
        gmm_norm_lr = gaussian_filter(gmm_norm_lr, sigma=max(0.5, sigma / 4.0))
    # 上采样到原图尺寸仅为视觉对齐
    gmm_heat = F.interpolate(
        torch.from_numpy(gmm_norm_lr).unsqueeze(0).unsqueeze(0).float(),
        size=(H_img, W_img), mode='bilinear', align_corners=False
    ).squeeze().numpy()

    # ============ 第二遍：窗口 GSUP ============
    all_up_logits, all_up_weights, all_positions = [], [], []

    with torch.no_grad():
        for item in window_cache:
            y, x = item['y'], item['x']
            y0, x0 = item['y0'], item['x0']
            py, px, h, w = item['py'], item['px'], item['h'], item['w']

            win_np = img_padded[y:y + win_size, x:x + win_size, :]
            win_tensor = preprocess_patch(win_np)

            # 从全局 z_map 裁剪（跨窗口一致的核心）
            z_crop = z_map[:, :, py:py + h, px:px + w]                # [1,C,h,w]
            logits_for_gsup = torch.log(z_crop.clamp(min=1e-6))

            if use_gsup:
                logits_lr = rearrange(logits_for_gsup[0], "c h w -> (h w) c").unsqueeze(0)
                coords_lr = create_coordinate_grid_2d(h, w, device).reshape(-1, 2).unsqueeze(0)
                coords_hr = create_coordinate_grid_2d(win_size, win_size, device).reshape(-1, 2).unsqueeze(0)
                image_lr = F.interpolate(win_tensor, size=(h, w),
                                         mode="bilinear", align_corners=False)
                pixels_lr = rearrange(image_lr, "b c h w -> b (h w) c")
                pixels_hr = rearrange(win_tensor, "b c h w -> b (h w) c")
                upsampler = GaussianFeatureUpsampler(
                    patch_coords_lr=coords_lr, patch_coords_hr=coords_hr,
                    pixels_lr=pixels_lr, pixels_hr=pixels_hr,
                ).to(device)
                up_logits = upsampler.forward(logits_lr)
                up_logits = up_logits.reshape(1, win_size, win_size, num_classes) \
                                     .permute(0, 3, 1, 2).squeeze(0)
            else:
                up_logits = F.interpolate(logits_for_gsup, size=(win_size, win_size),
                                          mode="bilinear", align_corners=False).squeeze(0)

            raw_weight = z_crop.max(dim=1)[0]
            raw_weight_hr = F.interpolate(raw_weight.unsqueeze(0),
                                          size=(win_size, win_size),
                                          mode="bilinear").squeeze(0).squeeze(0)

            all_up_logits.append(up_logits.cpu())
            all_up_weights.append(raw_weight_hr.cpu())
            all_positions.append((y0, x0))

    # ============ 跨窗口融合 ============
    acc_logits = torch.zeros((num_classes, H_img, W_img), dtype=torch.float32)
    acc_weights = torch.zeros((H_img, W_img), dtype=torch.float32)

    temperature = 0.07
    hann_1d = torch.hann_window(win_size)
    hann_2d = torch.outer(hann_1d, hann_1d)

    for i, (y0, x0) in enumerate(all_positions):
        exp_w = torch.exp(all_up_weights[i] / temperature) * hann_2d
        ys, ye = max(0, y0), min(H_img, y0 + win_size)
        xs, xe = max(0, x0), min(W_img, x0 + win_size)
        cy1, cy2 = ys - y0, ys - y0 + (ye - ys)
        cx1, cx2 = xs - x0, xs - x0 + (xe - xs)
        acc_logits[:, ys:ye, xs:xe] += all_up_logits[i][:, cy1:cy2, cx1:cx2] * \
                                       exp_w[cy1:cy2, cx1:cx2].unsqueeze(0)
        acc_weights[ys:ye, xs:xe] += exp_w[cy1:cy2, cx1:cx2]

    final_logits = acc_logits / acc_weights.clamp(min=1e-6)
    final_probs = F.softmax(final_logits, dim=0)

    # ============ GSUP 结果渲染 ============
    gsup_target = final_probs[target_idx].cpu().numpy()
    gsup_heat = _normalize_map(gsup_target, norm_mode, p_lo, p_hi)
    if sigma > 0:
        gsup_heat = gaussian_filter(gsup_heat, sigma=sigma)

    # ============ 保存三张图 + 三联图 ============
    Image.fromarray(img_np).save(f"{out_base}_input.png")
    Image.fromarray((PURPLE_YELLOW(gmm_heat)[..., :3] * 255).astype(np.uint8)).save(f"{out_base}_gmm.png")
    Image.fromarray((PURPLE_YELLOW(gsup_heat)[..., :3] * 255).astype(np.uint8)).save(f"{out_base}_gsup.png")

    tag_gmm = "GMM (global)" if use_gmm else "Raw text logits"
    tag_gsup = "GSUP (windowed)" if use_gsup else "Bilinear up"

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(image); axes[0].set_title(f"Input\n{class_groups[target_idx]}", fontsize=12)
    axes[0].axis("off")
    axes[1].imshow(gmm_heat, cmap=PURPLE_YELLOW, vmin=0, vmax=1)
    axes[1].set_title(f"{tag_gmm}\n(low-res patch grid → upsampled)", fontsize=12)
    axes[1].axis("off")
    axes[2].imshow(gsup_heat, cmap=PURPLE_YELLOW, vmin=0, vmax=1)
    axes[2].set_title(f"{tag_gsup}\n(full-res)", fontsize=12)
    axes[2].axis("off")
    plt.tight_layout()
    plt.savefig(f"{out_base}_compare.png", bbox_inches='tight', dpi=150)
    if show:
        plt.show()
    plt.close()

    # ============ 可选 mask ============
    if save_mask:
        mask_bin = (gsup_heat > mask_thresh).astype(np.uint8) * 255
        Image.fromarray(mask_bin).save(f"{out_base}_mask.png")
        overlay = img_np.copy()
        color = np.array([255, 235, 59], dtype=np.uint8)
        m = mask_bin > 0
        overlay[m] = (overlay[m] * 0.5 + color * 0.5).astype(np.uint8)
        Image.fromarray(overlay).save(f"{out_base}_mask_overlay.png")

    return gmm_heat, gsup_heat


def main():
    parser = argparse.ArgumentParser(
        description="DINOv3 attention: Input / Global-GMM / Windowed-GSUP (no TLP)."
    )
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", "-o")
    parser.add_argument("--class", "-c", dest="target_class", default="building")
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--norm", choices=["minmax", "percentile"], default="minmax")
    parser.add_argument("--p-lo", type=float, default=2.0)
    parser.add_argument("--p-hi", type=float, default=98.0)

    # 只剩两个开关
    parser.add_argument("--use-gmm", action="store_true", default=True,
                        help="Enable global GMM.")
    parser.add_argument("--no-gmm", action="store_false", dest="use_gmm")
    parser.add_argument("--use-gsup", action="store_true", default=True)
    parser.add_argument("--no-gsup", action="store_false", dest="use_gsup")

    parser.add_argument("--gmm-max-iter", type=int, default=10)
    parser.add_argument("--gmm-temp", type=float, default=50.0)
    parser.add_argument("--save-mask", action="store_true")
    parser.add_argument("--mask-thresh", type=float, default=0.5)
    args = parser.parse_args()

    target_idx = resolve_target_index(args.target_class)
    print(f"Target: '{args.target_class}' -> class_groups[{target_idx}]  |  "
          f"GMM={'ON' if args.use_gmm else 'OFF'}  |  "
          f"GSUP={'ON' if args.use_gsup else 'OFF'}  |  TLP=removed  |  "
          f"norm={args.norm}")

    def _make_out_base(in_file, out_arg):
        """统一决定 out_base（无扩展名的前缀路径）"""
        if out_arg is None:
            d = os.path.dirname(in_file) or "."
            name, _ = os.path.splitext(os.path.basename(in_file))
            return os.path.join(d, name)
        if os.path.isdir(out_arg):
            name, _ = os.path.splitext(os.path.basename(in_file))
            return os.path.join(out_arg, name)
        base, _ = os.path.splitext(out_arg)
        return base

    def _run(in_file, out_base, show):
        predict_attention(
            in_file, target_idx, out_base, show=show,
            sigma=args.sigma, norm_mode=args.norm,
            p_lo=args.p_lo, p_hi=args.p_hi,
            use_gsup=args.use_gsup, use_gmm=args.use_gmm,
            gmm_max_iter=args.gmm_max_iter, gmm_temp=args.gmm_temp,
            save_mask=args.save_mask, mask_thresh=args.mask_thresh,
        )

    if os.path.isfile(args.input):
        out_base = _make_out_base(args.input, args.output)
        _run(args.input, out_base, show=True)
        print(f"Saved: {out_base}_input.png / _gmm.png / _gsup.png / _compare.png")

    elif os.path.isdir(args.input):
        out_dir = args.output or args.input
        os.makedirs(out_dir, exist_ok=True)
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif")
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(args.input, ext)))
            files.extend(glob.glob(os.path.join(args.input, ext.upper())))
        # 过滤掉本脚本自己生成的文件，避免自吞
        files = [f for f in files if not any(
            os.path.basename(f).endswith(s) for s in
            ("_input.png", "_gmm.png", "_gsup.png", "_compare.png", "_mask.png", "_mask_overlay.png", "_attn.png")
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