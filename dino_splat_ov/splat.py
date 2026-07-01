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

from dino_splat_ov.dinov3.data.transforms import make_classification_eval_transform
from dino_splat_ov.dinov3.hub.dinotxt import dinov3_vitl16_dinotxt_tet1280d20h24l

# =========================
# Gaussian JBU
# =========================
from dino_splat_ov.gsup import GaussianFeatureUpsampler, create_coordinate_grid_2d

# =========================
# 1. Load model
# =========================
model, tokenizer = dinov3_vitl16_dinotxt_tet1280d20h24l()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device).eval()

# =========================
# 2. 输入图像和类别分组
# =========================
image_path = "asset/img.jpg"
image = Image.open(image_path).convert("RGB")
orig_w, orig_h = image.size
H_img, W_img = orig_h, orig_w

class_groups = [
    ["bareland", "barren"],
    ["pavement"],
    ["road"],
    ["forest", "tree"],
    ["river", "water"],
    ["grass"],
    ["field"],
    ["building", "house", "roof"],
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
num_flat = len(flat_texts)

# =========================
# 3. Sliding window preparation
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

img_np = np.array(image)
img_padded, pad_top, pad_left = pad_to_multiple(img_np, win_size, stride)
H_pad, W_pad = img_padded.shape[:2]

windows = []
for y in range(0, H_pad - win_size + 1, stride):
    for x in range(0, W_pad - win_size + 1, stride):
        win = img_padded[y : y + win_size, x : x + win_size, :]
        windows.append((win, y - pad_top, x - pad_left))

# =========================
# 4. Preprocessing
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
# 6. 收集所有窗口的数据（不累加）
# =========================
all_img_feats = []      # 存储低分辨率特征 [1, D, h, w]
all_up_diffs = []       # 存储上采样后的差异图 [num_classes, 224, 224]  <-- 修改
all_positions = []      # (y0, x0)

with torch.no_grad():
    for win_np, y0, x0 in windows:
        if y0 >= H_img or x0 >= W_img or y0 + win_size <= 0 or x0 + win_size <= 0:
            continue

        win_tensor = preprocess_patch(win_np)

        # DINOv3 forward
        _, image_patch_tokens, _ = model.encode_image_with_patch_tokens(win_tensor)
        B, P, D = image_patch_tokens.shape
        h = w = int(P**0.5)
        img_feat = image_patch_tokens.transpose(1, 2).reshape(B, D, h, w)

        # 文本特征
        text_feat = model.encode_text(text_tokens)[:, 1024:]  # [num_flat, D]
        img_feat = F.normalize(img_feat, dim=1)
        text_feat = F.normalize(text_feat, dim=-1)

        # 独立文本 logits
        logits_flat = torch.einsum("bchw,nc->bnhw", img_feat, text_feat)  # [1, num_flat, h, w]

        # 同义词合并
        merged_logits = []
        for idx_list in group_index_maps:
            weights = [1.0] + [0.3] * (len(idx_list) - 1)
            weighted_sum = torch.zeros_like(logits_flat[:, idx_list[0], :, :])
            for i, idx in enumerate(idx_list):
                weighted_sum += logits_flat[:, idx, :, :] * weights[i]
            group_logit = weighted_sum / sum(weights)
            merged_logits.append(group_logit.unsqueeze(1))
        logits = torch.cat(merged_logits, dim=1)  # [1, num_classes, h, w]

        # ========== 核心修改：计算差异图（最优路径 / Wasserstein 简化版） ==========
        # 1) 将 logits 转为概率分布（温度=1.0）
        prob = torch.softmax(logits, dim=1)  # [1, num_classes, h, w]
        # 2) 均匀分布
        uniform = 1.0 / num_classes
        # 3) 每个类别的偏差绝对值（即与均匀分布的距离）
        diff = torch.clamp(prob - uniform, min=0)
        # 现在 diff 替代 prob 作为后续上采样的输入
        # ========================================================================

        # Gaussian JBU upsampling（用 diff 代替 prob）
        diff_lr = rearrange(diff[0], "c h w -> (h w) c").unsqueeze(0)  # [1, h*w, num_classes]
        patch_coords_lr = create_coordinate_grid_2d(h, w, device).reshape(-1, 2).unsqueeze(0)
        patch_coords_hr = create_coordinate_grid_2d(win_size, win_size, device).reshape(-1, 2).unsqueeze(0)

        image_lr = F.interpolate(win_tensor, size=(h, w), mode="bilinear", align_corners=False)
        pixels_lr = rearrange(image_lr, "b c h w -> b (h w) c")
        pixels_hr = rearrange(win_tensor, "b c h w -> b (h w) c")

        upsampler = GaussianFeatureUpsampler(
            patch_coords_lr=patch_coords_lr,
            patch_coords_hr=patch_coords_hr,
            pixels_lr=pixels_lr,
            pixels_hr=pixels_hr,
        ).to(device)
        upsampled = upsampler.forward(diff_lr)  # [1, win_size*win_size, num_classes]
        up_diff = upsampled.reshape(1, win_size, win_size, num_classes).permute(0, 3, 1, 2)  # [1, num_classes, 224, 224]
        up_diff = up_diff.squeeze(0)  # [num_classes, 224, 224]

        # 存储
        all_img_feats.append(img_feat.cpu())
        all_up_diffs.append(up_diff.cpu())      # 修改变量名
        all_positions.append((y0, x0))

# =========================
# 7. 计算跨窗口注意力权重（GLA核心）
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
all_img_feats = [f.to(device) for f in all_img_feats]
all_img_feats = torch.cat(all_img_feats, dim=0)  # [N, D, h, w]
N, D, h, w = all_img_feats.shape

# 全局锚点：所有窗口所有像素的平均
global_anchor = all_img_feats.mean(dim=(0, 2, 3), keepdim=True)  # [1, D, 1, 1]

# 展平每个窗口的特征为 Key
keys = all_img_feats.flatten(start_dim=2)  # [N, D, h*w]
scores = torch.einsum('d, n d l -> n l', global_anchor.squeeze(), keys)  # [N, h*w]
attn = F.softmax(scores / 0.07, dim=0)  # [N, h*w]  对窗口维度 softmax

# 重塑为 [N, 1, h, w]
attn_maps = attn.reshape(N, 1, h, w)  # [N, 1, h, w]
# 上采样到 224x224
attn_maps_224 = F.interpolate(attn_maps, size=(win_size, win_size), mode='bilinear')  # [N, 1, 224, 224]

# =========================
# 8. 用注意力权重重新累加融合（使用差异图）
# =========================
acc_diffs = torch.zeros((num_classes, H_img, W_img), dtype=torch.float32, device=device)   # 修改
acc_weights = torch.zeros((H_img, W_img), dtype=torch.float32, device=device)

for idx, (y0, x0) in enumerate(all_positions):
    up_diff = all_up_diffs[idx].to(device)          # [num_classes, 224, 224]
    attn_map = attn_maps_224[idx]                   # [1, 224, 224]

    # 裁剪到原图有效区域
    y_start = max(0, y0)
    y_end = min(H_img, y0 + win_size)
    x_start = max(0, x0)
    x_end = min(W_img, x0 + win_size)
    crop_y1 = y_start - y0
    crop_y2 = crop_y1 + (y_end - y_start)
    crop_x1 = x_start - x0
    crop_x2 = crop_x1 + (x_end - x_start)

    win_diff_crop = up_diff[:, crop_y1:crop_y2, crop_x1:crop_x2]
    win_weight_crop = attn_map[:, crop_y1:crop_y2, crop_x1:crop_x2]  # [1, Hc, Wc]

    acc_diffs[:, y_start:y_end, x_start:x_end] += win_diff_crop * win_weight_crop
    acc_weights[y_start:y_end, x_start:x_end] += win_weight_crop.squeeze(0)

# 归一化
acc_weights = acc_weights.clamp(min=1e-6)
final_diff = acc_diffs / acc_weights  # [num_classes, H, W]  修改
final_diff = final_diff.unsqueeze(0)  # [1, num_classes, H, W]

# 深度可分离高斯平滑（保持与原代码一致）
kernel = (
    torch.tensor(
        [[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32, device=device
    ).view(1, 1, 3, 3)
    / 16
)
kernel = kernel.repeat(num_classes, 1, 1, 1)
final_diff_smooth = F.conv2d(
    final_diff, weight=kernel, padding=1, groups=num_classes
)

final_diff_np = final_diff_smooth.squeeze(0).cpu().numpy()  # [num_classes, H, W]
# 直接取差异最大的类别（偏差最大的类别）
mask = np.argmax(final_diff_np, axis=0)   # 修改：取最大值
mask = median_filter(mask, size=3)

# =========================
# 9. 可视化
# =========================
preset_palette = [
    (72, 40, 120), (62, 74, 137), (49, 104, 142), (38, 130, 142),
    (31, 158, 137), (73, 193, 110), (160, 218, 57), (253, 231, 37),
]
preset_palette = np.array(preset_palette) / 255.0
if num_classes <= len(preset_palette):
    palette = preset_palette[:num_classes]
else:
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
plt.title("GLA + Ours (diff-based)")
plt.axis("off")
plt.show()