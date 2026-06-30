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

# =========================
# DINOv3
# =========================
root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

from dinov3.data.transforms import make_classification_eval_transform
from dinov3.hub.dinotxt import dinov3_vitl16_dinotxt_tet1280d20h24l

# =========================
# Gaussian JBU
# =========================
from splatov.gsup import GaussianFeatureUpsampler, create_coordinate_grid_2d

# =========================
# 1. Load model
# =========================
model, tokenizer = dinov3_vitl16_dinotxt_tet1280d20h24l()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device).eval()

# =========================
# 2. 输入图像和类别分组（支持同义词）
# =========================
image_path = "asset/img.jpg"
image = Image.open(image_path).convert("RGB")
orig_w, orig_h = image.size
H_img, W_img = orig_h, orig_w

# -------------------- 修改开始 --------------------
# 定义同义词分组：每个子列表代表一个语义类别
class_groups = [
    # ["background"],
    ["bareland", "barren"],
    ["pavement"],
    ["road"],
    ["forest", "tree"],
    ["river", "water"],
    ["grass"],
    ["field"],
    ["building", "house", "roof"],  # 主词 building，同义词 house, roof
]

# 展平所有文本，用于编码
flat_texts = []
group_index_maps = []  # 记录每个组对应 flat_texts 中的索引列表
for group in class_groups:
    start = len(flat_texts)
    flat_texts.extend(group)
    end = len(flat_texts)
    group_index_maps.append(list(range(start, end)))

texts = [f"a photo of {c}" for c in flat_texts]
num_classes = len(class_groups)  # 最终输出的类别数（合并后）
num_flat = len(flat_texts)  # 原始文本总数（含同义词）
# -------------------- 修改结束 --------------------

# =========================
# 3. Sliding window preparation
# =========================
win_size = 224
stride = 112


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


img_np = np.array(image)
img_padded, pad_top, pad_left = pad_to_multiple(img_np, win_size, stride)
H_pad, W_pad = img_padded.shape[:2]

windows = []
for y in range(0, H_pad - win_size + 1, stride):
    for x in range(0, W_pad - win_size + 1, stride):
        win = img_padded[y : y + win_size, x : x + win_size, :]
        windows.append((win, y - pad_top, x - pad_left))

# =========================
# 4. Preprocessing function for patches
# =========================
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize = Normalize(mean, std)
to_tensor = ToTensor()


def preprocess_patch(patch_np):
    patch_pil = Image.fromarray(patch_np)
    tensor = to_tensor(patch_pil)
    tensor = normalize(tensor)
    return tensor.unsqueeze(0).to(device)


# =========================
# 5. Tokenize all flattened texts globally
# =========================
text_tokens = tokenizer.tokenize(texts).to(device)

# =========================
# 6. Accumulators on GPU (shape: [num_classes, H, W])
# =========================
acc_logits = torch.zeros(
    (num_classes, H_img, W_img), dtype=torch.float32, device=device
)
acc_weights = torch.zeros((H_img, W_img), dtype=torch.float32, device=device)


# Gaussian weight for window fusion (centered)
def get_gaussian_weight(size, sigma=0.5):
    ax = torch.linspace(-1, 1, size, device=device)
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    w = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return w


gauss_weight = get_gaussian_weight(win_size, sigma=0.5)

# =========================
# 7. Sliding window processing
# =========================
with torch.no_grad():
    for win_np, y0, x0 in windows:
        # Skip windows entirely outside original image
        if y0 >= H_img or x0 >= W_img or y0 + win_size <= 0 or x0 + win_size <= 0:
            continue

        win_tensor = preprocess_patch(win_np)

        # DINOv3 forward
        _, image_patch_tokens, _ = model.encode_image_with_patch_tokens(win_tensor)
        B, P, D = image_patch_tokens.shape
        h = w = int(P**0.5)
        img_feat = image_patch_tokens.transpose(1, 2).reshape(B, D, h, w)

        # 编码所有展平后的文本（含同义词）
        text_feat = model.encode_text(text_tokens)[:, 1024:]  # [num_flat, D]
        img_feat = F.normalize(img_feat, dim=1)
        text_feat = F.normalize(text_feat, dim=-1)

        # 1. 先计算所有独立文本的 Logits
        logits_flat = torch.einsum(
            "bchw,nc->bnhw", img_feat, text_feat
        )  # [1, num_flat, h, w]

        # ====================== 修改开始（加权合并同义词） ======================
        # 2. 根据同义词分组合并 Logits（加权平均，主词权重大）
        merged_logits = []
        for idx_list in group_index_maps:
            # 第一个索引为主词，其余为同义词
            # 权重：主词 1.0，同义词 0.3（可调）
            weights = [1.0] + [0.3] * (len(idx_list) - 1)
            # 计算加权和
            weighted_sum = torch.zeros_like(logits_flat[:, idx_list[0], :, :])
            for i, idx in enumerate(idx_list):
                weighted_sum += logits_flat[:, idx, :, :] * weights[i]
            # 归一化权重，保持量级不变
            group_logit = weighted_sum / sum(weights)
            merged_logits.append(group_logit.unsqueeze(1))  # [1, 1, h, w]
        logits = torch.cat(merged_logits, dim=1)  # [1, num_classes, h, w]
        # ====================== 修改结束 ======================

        # 3. 计算概率（此时类别数 = num_classes）
        prob = torch.softmax(logits / 0.07, dim=1)  # [1, num_classes, h, w]

        # Gaussian JBU upsampling to win_size
        prob_lr = rearrange(prob[0], "c h w -> (h w) c").unsqueeze(
            0
        )  # [1, h*w, num_classes]
        patch_coords_lr = (
            create_coordinate_grid_2d(h, w, device).reshape(-1, 2).unsqueeze(0)
        )
        patch_coords_hr = (
            create_coordinate_grid_2d(win_size, win_size, device)
            .reshape(-1, 2)
            .unsqueeze(0)
        )

        # RGB guidance (low-res and high-res)
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
        upsampled = upsampler.forward(prob_lr)  # [1, win_size*win_size, num_classes]
        up_prob = upsampled.reshape(1, win_size, win_size, num_classes).permute(
            0, 3, 1, 2
        )  # [1, num_classes, 224, 224]
        up_prob = up_prob.squeeze(0)  # [num_classes, 224, 224]

        # Crop overlapping region in original image coordinates
        y_start = max(0, y0)
        y_end = min(H_img, y0 + win_size)
        x_start = max(0, x0)
        x_end = min(W_img, x0 + win_size)
        if y_start >= y_end or x_start >= x_end:
            continue

        crop_y1 = y_start - y0
        crop_y2 = crop_y1 + (y_end - y_start)
        crop_x1 = x_start - x0
        crop_x2 = crop_x1 + (x_end - x_start)

        win_prob_crop = up_prob[:, crop_y1:crop_y2, crop_x1:crop_x2]
        win_weight_crop = gauss_weight[crop_y1:crop_y2, crop_x1:crop_x2]

        # Weighted accumulation
        acc_logits[:, y_start:y_end, x_start:x_end] += win_prob_crop * win_weight_crop
        acc_weights[y_start:y_end, x_start:x_end] += win_weight_crop

# =========================
# 8. Final logits and smoothing
# =========================
acc_weights = acc_weights.clamp(min=1e-6)
final_logits = acc_logits / acc_weights  # [num_classes, H, W]
final_logits = final_logits.unsqueeze(0)  # [1, num_classes, H, W]

# Depthwise Gaussian smoothing kernel (3x3)
kernel = (
    torch.tensor(
        [[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32, device=device
    ).view(1, 1, 3, 3)
    / 16
)
kernel = kernel.repeat(num_classes, 1, 1, 1)  # [num_classes, 1, 3, 3]
final_logits_smooth = F.conv2d(
    final_logits, weight=kernel, padding=1, groups=num_classes
)

# Convert to numpy and argmax
final_logits_np = final_logits_smooth.squeeze(0).cpu().numpy()  # [num_classes, H, W]
mask = np.argmax(final_logits_np, axis=0)  # [H, W]

# Optional median filter to remove small isolated blobs
mask = median_filter(mask, size=3)

# =========================
# 9. Visualization (自动适配类别数)
# =========================
# 预定义调色板（8种颜色，可自行扩展）
preset_palette = [
    (72, 40, 120),
    (62, 74, 137),
    (49, 104, 142),
    (38, 130, 142),
    (31, 158, 137),
    (73, 193, 110),
    (160, 218, 57),
    (253, 231, 37),
]
preset_palette = np.array(preset_palette) / 255.0

if num_classes <= len(preset_palette):
    palette = preset_palette[:num_classes]
else:
    # 如果类别数超过预设，从 matplotlib 颜色映射中取色
    cmap = plt.cm.get_cmap("tab20", num_classes)
    palette = np.array([cmap(i)[:3] for i in range(num_classes)])

cmap = ListedColormap(palette)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Input")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(mask, cmap=cmap, vmin=0, vmax=num_classes - 1)
plt.title("Sliding Window + Gaussian Splatting")
plt.axis("off")

plt.show()
