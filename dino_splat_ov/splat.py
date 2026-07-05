from pathlib import Path
import sys
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from einops import rearrange
import numpy as np
from matplotlib.colors import ListedColormap
from torchvision.transforms import Normalize, ToTensor
from scipy.ndimage import median_filter
import argparse
import os
import glob


# =========================
# 图拉普拉斯平滑（LPOSS）
# =========================
def graph_laplacian_smooth(
    logits: torch.Tensor,
    features: torch.Tensor,
    num_iter: int = 3,
    alpha: float = 0.6,
    temperature: float = 0.1,
    kernel_size: int = 3,
) -> torch.Tensor:
    C, H, W = logits.shape
    pad = kernel_size // 2
    feat_pad = F.pad(
        features.unsqueeze(0), (pad, pad, pad, pad), mode="reflect"
    ).squeeze(0)
    feat_unfold = F.unfold(feat_pad.unsqueeze(0), kernel_size=kernel_size, padding=0)
    feat_unfold = feat_unfold.reshape(
        features.shape[0], kernel_size * kernel_size, H, W
    )
    similarity = torch.einsum("dnhw,dhw->nhw", feat_unfold, features)
    affinity = torch.exp(similarity / temperature)
    affinity = affinity / affinity.sum(dim=0, keepdim=True)
    current_logits = logits.clone()
    for _ in range(num_iter):
        current_pad = F.pad(
            current_logits.unsqueeze(0), (pad, pad, pad, pad), mode="reflect"
        ).squeeze(0)
        current_unfold = F.unfold(
            current_pad.unsqueeze(0), kernel_size=kernel_size, padding=0
        )
        current_unfold = current_unfold.reshape(C, kernel_size * kernel_size, H, W)
        neighbor_avg = torch.einsum("cnhw,nhw->chw", current_unfold, affinity)
        current_logits = alpha * current_logits + (1 - alpha) * neighbor_avg
    return current_logits


# =========================
# 加载 DINOv3 模型（全局单例）
# =========================
root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

from dino_splat_ov.dinov3.data.transforms import make_classification_eval_transform
from dino_splat_ov.dinov3.hub.dinotxt import dinov3_vitl16_dinotxt_tet1280d20h24l
from dino_splat_ov.gsup import GaussianFeatureUpsampler, create_coordinate_grid_2d

model, tokenizer = dinov3_vitl16_dinotxt_tet1280d20h24l()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device).eval()

# =========================
# 类别分组（可自定义）
# =========================
class_groups = [
    ["pavement", "bareland", "barren"],
    ["road"],
    ["forest", "tree"],
    ["river", "water"],
    ["grass"],
    ["field", "cropland"],
    ["building", "house", "roof"],
]

# 彩色掩膜
preset_palette = [
    # (72, 40, 120),
    (62, 74, 137),
    (49, 104, 142),
    (38, 130, 142),
    (31, 158, 137),
    (73, 193, 110),
    (160, 218, 57),
    (253, 231, 37),
]

flat_texts = []
group_index_maps = []
for group in class_groups:
    start = len(flat_texts)
    flat_texts.extend(group)
    end = len(flat_texts)
    group_index_maps.append(list(range(start, end)))

texts = [f"a photo of {c}" for c in flat_texts]
num_classes = len(class_groups)

# =========================
# 滑动窗口参数
# =========================
win_size = 256
stride = 128


def pad_to_multiple(img, win_size, stride):
    h, w = img.shape[:2]
    pad_h = (win_size - h) % stride if h < win_size else (win_size - h) % stride
    pad_w = (win_size - w) % stride if w < win_size else (win_size - w) % stride
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    img_padded = np.pad(
        img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode="reflect"
    )
    return img_padded, pad_top, pad_left


def preprocess_patch(patch_np):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = Normalize(mean, std)
    to_tensor = ToTensor()
    patch_pil = Image.fromarray(patch_np)
    tensor = to_tensor(patch_pil)
    tensor = normalize(tensor)
    return tensor.unsqueeze(0).to(device)


# =========================
# 核心预测函数
# =========================
def predict_image(image_path, output_path=None, show=False):
    """
    对单张图像进行预测，保存结果，并可选择可视化显示。
    Args:
        image_path: 输入图像路径
        output_path: 输出路径（若为None则自动生成）
        show: 是否显示可视化窗口
    Returns:
        mask: 预测标签图 (H, W) numpy数组
    """
    # 读取图像
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size
    H_img, W_img = orig_h, orig_w
    img_np = np.array(image)

    # 文本特征（全局一次）
    with torch.no_grad():
        text_tokens = tokenizer.tokenize(texts).to(device)

    # 滑动窗口
    img_padded, pad_top, pad_left = pad_to_multiple(img_np, win_size, stride)
    H_pad, W_pad = img_padded.shape[:2]

    windows = []
    for y in range(0, H_pad - win_size + 1, stride):
        for x in range(0, W_pad - win_size + 1, stride):
            win = img_padded[y : y + win_size, x : x + win_size, :]
            windows.append((win, y - pad_top, x - pad_left))

    all_up_logits = []
    all_up_weights = []
    all_positions = []

    with torch.no_grad():
        for win_np, y0, x0 in windows:
            if y0 >= H_img or x0 >= W_img or y0 + win_size <= 0 or x0 + win_size <= 0:
                continue

            win_tensor = preprocess_patch(win_np)

            _, image_patch_tokens, _ = model.encode_image_with_patch_tokens(win_tensor)
            B, P, D = image_patch_tokens.shape
            h = w = int(P**0.5)
            img_feat = image_patch_tokens.transpose(1, 2).reshape(B, D, h, w)

            text_feat = model.encode_text(text_tokens)[:, 1024:]  # [num_flat, D]
            img_feat = F.normalize(img_feat, dim=1)
            text_feat = F.normalize(text_feat, dim=-1)

            logits_flat = torch.einsum("bchw,nc->bnhw", img_feat, text_feat)
            raw_weight = logits_flat.max(dim=1)[0]

            merged_logits = []
            for idx_list in group_index_maps:
                weights = [1.0] + [0.3] * (len(idx_list) - 1)
                weighted_sum = torch.zeros_like(logits_flat[:, idx_list[0], :, :])
                for i, idx in enumerate(idx_list):
                    weighted_sum += logits_flat[:, idx, :, :] * weights[i]
                group_logit = weighted_sum / sum(weights)
                merged_logits.append(group_logit.unsqueeze(1))
            logits = torch.cat(merged_logits, dim=1)

            # 图拉普拉斯平滑
            logits_smooth = graph_laplacian_smooth(
                logits=logits.squeeze(0),
                features=img_feat.squeeze(0),
                num_iter=3,
                alpha=0.6,
                temperature=0.1,
            )
            logits = logits_smooth.unsqueeze(0)

            # JBU 上采样
            logits_lr = rearrange(logits[0], "c h w -> (h w) c").unsqueeze(0)
            patch_coords_lr = (
                create_coordinate_grid_2d(h, w, device).reshape(-1, 2).unsqueeze(0)
            )
            patch_coords_hr = (
                create_coordinate_grid_2d(win_size, win_size, device)
                .reshape(-1, 2)
                .unsqueeze(0)
            )

            image_lr = F.interpolate(
                win_tensor, size=(h, w), mode="bilinear", align_corners=False
            )
            pixels_lr = rearrange(image_lr, "b c h w -> b (h w) c")
            pixels_hr = rearrange(win_tensor, "b c h w -> b (h w) c")

            upsampler = GaussianFeatureUpsampler(
                patch_coords_lr=patch_coords_lr,
                patch_coords_hr=patch_coords_hr,
                pixels_lr=pixels_lr,
                pixels_hr=pixels_hr,
            ).to(device)
            up_logits = upsampler.forward(logits_lr)
            up_logits = (
                up_logits.reshape(1, win_size, win_size, num_classes)
                .permute(0, 3, 1, 2)
                .squeeze(0)
            )

            raw_weight_hr = F.interpolate(
                raw_weight.unsqueeze(0), size=(win_size, win_size), mode="bilinear"
            ).squeeze(0)
            raw_weight_hr = raw_weight_hr.squeeze(0)

            all_up_logits.append(up_logits.cpu())
            all_up_weights.append(raw_weight_hr.cpu())
            all_positions.append((y0, x0))

    # 跨窗口融合
    device_cpu = torch.device("cpu")
    acc_weighted_logits = torch.zeros(
        (num_classes, H_img, W_img), dtype=torch.float32, device=device_cpu
    )
    acc_exp_weights = torch.zeros(
        (H_img, W_img), dtype=torch.float32, device=device_cpu
    )

    temperature = 0.07
    hann_1d = torch.hann_window(win_size, device=device_cpu)
    hann_2d = torch.outer(hann_1d, hann_1d)

    for idx, (y0, x0) in enumerate(all_positions):
        up_logits = all_up_logits[idx]
        up_weight = all_up_weights[idx]

        exp_w = torch.exp(up_weight / temperature)
        exp_w = exp_w * hann_2d

        y_start = max(0, y0)
        y_end = min(H_img, y0 + win_size)
        x_start = max(0, x0)
        x_end = min(W_img, x0 + win_size)
        crop_y1 = y_start - y0
        crop_y2 = crop_y1 + (y_end - y_start)
        crop_x1 = x_start - x0
        crop_x2 = crop_x1 + (x_end - x_start)

        logits_crop = up_logits[:, crop_y1:crop_y2, crop_x1:crop_x2]
        exp_w_crop = exp_w[crop_y1:crop_y2, crop_x1:crop_x2]

        acc_weighted_logits[:, y_start:y_end, x_start:x_end] += (
            logits_crop * exp_w_crop.unsqueeze(0)
        )
        acc_exp_weights[y_start:y_end, x_start:x_end] += exp_w_crop

    final_logits = acc_weighted_logits / (acc_exp_weights.clamp(min=1e-6))
    final_probs = F.softmax(final_logits, dim=0)
    final_probs_np = final_probs.cpu().numpy()
    mask = np.argmax(final_probs_np, axis=0)
    mask = median_filter(mask, size=3)

    # ---------- 保存结果 ----------
    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = base + "_ours.png"

    preset_palette2 = np.array(preset_palette)
    if num_classes <= len(preset_palette2):
        palette = preset_palette2[:num_classes]
    else:
        cmap = plt.cm.get_cmap("tab20", num_classes)
        palette = np.array(
            [np.array(cmap(i)[:3]) * 255 for i in range(num_classes)], dtype=np.uint8
        )

    mask_color = np.zeros((H_img, W_img, 3), dtype=np.uint8)
    for class_id in range(num_classes):
        mask_color[mask == class_id] = palette[class_id]

    Image.fromarray(mask_color).save(output_path)
    # 同时保存灰度标签图
    # label_path = os.path.splitext(output_path)[0] + "_label.png"
    # Image.fromarray(mask.astype(np.uint8)).save(label_path)

    # ---------- 可视化（单文件模式） ----------
    if show:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title("Input")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        # 使用调色板显示
        cmap_display = ListedColormap(np.array(palette) / 255.0)
        plt.imshow(mask, cmap=cmap_display, vmin=0, vmax=num_classes - 1)
        plt.title("Segmentation Mask")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return mask


# =========================
# 命令行入口
# =========================
def main():
    parser = argparse.ArgumentParser(
        description="DINOv3-based semantic segmentation with text prior."
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to an image file or a directory containing images.",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output path (file or directory). If not specified, auto-generate.",
    )
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output

    if os.path.isfile(input_path):
        # 单张图片 → 显示可视化
        if output_path is None:
            output_path = None
        else:
            if os.path.isdir(output_path):
                base = os.path.basename(input_path)
                name, _ = os.path.splitext(base)
                output_path = os.path.join(output_path, name + "_ours.png")
        predict_image(input_path, output_path, show=True)
        print(f"Processed {input_path} -> {output_path}")

    elif os.path.isdir(input_path):
        # 批量处理 → 不显示可视化
        if output_path is None:
            output_path = input_path
        os.makedirs(output_path, exist_ok=True)

        extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff")
        image_files = []
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(input_path, ext)))
            image_files.extend(glob.glob(os.path.join(input_path, ext.upper())))

        if not image_files:
            print(f"No image files found in {input_path}")
            return

        for img_file in image_files:
            base = os.path.basename(img_file)
            name, _ = os.path.splitext(base)
            out_file = os.path.join(output_path, name + "_ours.png")
            predict_image(img_file, out_file, show=False)
            print(f"Processed {img_file} -> {out_file}")
    else:
        print(f"Input path {input_path} does not exist.")
        return


if __name__ == "__main__":
    main()
