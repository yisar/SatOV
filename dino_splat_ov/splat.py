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
# 新增：基于DINO亲和力的图拉普拉斯平滑（LPOSS思路）
# =========================
def graph_laplacian_smooth(
    logits: torch.Tensor,
    features: torch.Tensor,
    num_iter: int = 3,
    alpha: float = 0.6,
    temperature: float = 0.1,
    kernel_size: int = 3
) -> torch.Tensor:
    """
    基于DINO视觉特征亲和力的迭代式图拉普拉斯平滑（等价于标签传播）
    对应LPOSS中用DINO特征做语义正则化的核心逻辑，仅在低分辨率patch阶段执行
    Args:
        logits: 初始分割logits [C, H, W]
        features: DINO归一化视觉特征 [D, H, W]
        num_iter: 平滑迭代次数
        alpha: 原始logits保留权重，越大平滑越弱
        temperature: 相似度温度，越小边缘保留越好
        kernel_size: 局部邻域大小，默认3即8邻域
    Returns:
        smoothed_logits: 平滑后logits [C, H, W]
    """
    C, H, W = logits.shape
    pad = kernel_size // 2

    # 镜像填充避免边缘失真
    feat_pad = F.pad(features.unsqueeze(0), (pad, pad, pad, pad), mode="reflect").squeeze(0)
    
    # 提取邻域特征块 [D, K², H, W]
    feat_unfold = F.unfold(feat_pad.unsqueeze(0), kernel_size=kernel_size, padding=0)
    feat_unfold = feat_unfold.reshape(features.shape[0], kernel_size * kernel_size, H, W)

    # 计算中心-邻域余弦相似度（特征已归一化，点积即余弦）
    similarity = torch.einsum("dnhw,dhw->nhw", feat_unfold, features)  # [K², H, W]
    
    # 温度缩放得到亲和力权重，行归一化
    affinity = torch.exp(similarity / temperature)
    affinity = affinity / affinity.sum(dim=0, keepdim=True)  # [K², H, W]

    # 迭代式拉普拉斯平滑
    current_logits = logits.clone()
    for _ in range(num_iter):
        current_pad = F.pad(current_logits.unsqueeze(0), (pad, pad, pad, pad), mode="reflect").squeeze(0)
        current_unfold = F.unfold(current_pad.unsqueeze(0), kernel_size=kernel_size, padding=0)
        current_unfold = current_unfold.reshape(C, kernel_size * kernel_size, H, W)

        # 邻域加权平均
        neighbor_avg = torch.einsum("cnhw,nhw->chw", current_unfold, affinity)
        # 更新：保留原始信息 + 邻域平滑
        current_logits = alpha * current_logits + (1 - alpha) * neighbor_avg

    return current_logits

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
    ["field", "cropland"],
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
# 6. 收集所有窗口的数据（存储 logits 和权重）
# =========================
all_up_logits = []      # 存储每个窗口上采样后的 logits [num_classes, win_size, win_size]
all_up_weights = []     # 存储每个窗口的权重图（max相似度上采样） [win_size, win_size]
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

        # 独立文本相似度 (logits)
        logits_flat = torch.einsum("bchw,nc->bnhw", img_feat, text_feat)  # [1, num_flat, h, w]

        # --- 方案2：计算原始权重（每个像素取最大文本相似度） ---
        raw_weight = logits_flat.max(dim=1)[0]  # [1, h, w]

        # 同义词合并，得到类别 logits（未除以温度）
        merged_logits = []
        for idx_list in group_index_maps:
            weights = [1.0] + [0.3] * (len(idx_list) - 1)
            weighted_sum = torch.zeros_like(logits_flat[:, idx_list[0], :, :])
            for i, idx in enumerate(idx_list):
                weighted_sum += logits_flat[:, idx, :, :] * weights[i]
            group_logit = weighted_sum / sum(weights)
            merged_logits.append(group_logit.unsqueeze(1))
        logits = torch.cat(merged_logits, dim=1)  # [1, num_classes, h, w]

        # =========================
        # 新增：DINO亲和力图拉普拉斯平滑（LPOSS思路）
        # =========================
        logits_smooth = graph_laplacian_smooth(
            logits=logits.squeeze(0),      # [num_classes, h, w]
            features=img_feat.squeeze(0),  # [D, h, w] 复用DINOv3原生patch特征
            num_iter=3,
            alpha=0.6,
            temperature=0.1
        )
        logits = logits_smooth.unsqueeze(0)  # 恢复 [1, num_classes, h, w] 维度，后续逻辑完全不变

        # --- 方案4：用 Gaussian JBU 上采样 logits ---
        logits_lr = rearrange(logits[0], "c h w -> (h w) c").unsqueeze(0)  # [1, h*w, num_classes]
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
        up_logits = upsampler.forward(logits_lr)  # [1, win_size*win_size, num_classes]
        up_logits = up_logits.reshape(1, win_size, win_size, num_classes).permute(0, 3, 1, 2).squeeze(0)  # [num_classes, win_size, win_size]

        # 上采样权重图（双线性插值）
        raw_weight_hr = F.interpolate(raw_weight.unsqueeze(0), size=(win_size, win_size), mode='bilinear').squeeze(0)  # [1, win_size, win_size]
        raw_weight_hr = raw_weight_hr.squeeze(0)  # [win_size, win_size]

        # 存储
        all_up_logits.append(up_logits.cpu())
        all_up_weights.append(raw_weight_hr.cpu())
        all_positions.append((y0, x0))

# =========================
# 7. 基于文本先验的跨窗口融合（Softmax 归一化权重）
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
acc_weighted_logits = torch.zeros((num_classes, H_img, W_img), dtype=torch.float32, device=device)
acc_exp_weights = torch.zeros((H_img, W_img), dtype=torch.float32, device=device)

temperature = 0.07  # 可调，控制权重分布的尖锐程度

# ---- 新增：生成二维汉宁窗 ----
hann_1d = torch.hann_window(win_size, device=device)          # [win_size]
hann_2d = torch.outer(hann_1d, hann_1d)                      # [win_size, win_size]
# ------------------------------

for idx, (y0, x0) in enumerate(all_positions):
    up_logits = all_up_logits[idx].to(device)          # [num_classes, 224, 224]
    up_weight = all_up_weights[idx].to(device)         # [224, 224]

    # 计算 exp(weight / temperature)
    exp_w = torch.exp(up_weight / temperature)
    # ---- 新增：乘以汉宁窗（空间平滑） ----
    exp_w = exp_w * hann_2d
    # ------------------------------

    # 裁剪到原图有效区域
    y_start = max(0, y0)
    y_end = min(H_img, y0 + win_size)
    x_start = max(0, x0)
    x_end = min(W_img, x0 + win_size)
    crop_y1 = y_start - y0
    crop_y2 = crop_y1 + (y_end - y_start)
    crop_x1 = x_start - x0
    crop_x2 = crop_x1 + (x_end - x_start)

    logits_crop = up_logits[:, crop_y1:crop_y2, crop_x1:crop_x2]
    exp_w_crop = exp_w[crop_y1:crop_y2, crop_x1:crop_x2]  # [Hc, Wc]

    # 累加加权 logits 和 exp(weight) 总和
    acc_weighted_logits[:, y_start:y_end, x_start:x_end] += logits_crop * exp_w_crop.unsqueeze(0)
    acc_exp_weights[y_start:y_end, x_start:x_end] += exp_w_crop

# 归一化得到最终 logits
final_logits = acc_weighted_logits / (acc_exp_weights.clamp(min=1e-6))  # [num_classes, H, W]


# 转为概率并生成掩膜
final_probs = F.softmax(final_logits, dim=0)  # [num_classes, H, W]
final_probs_np = final_probs.cpu().numpy()
mask = np.argmax(final_probs_np, axis=0)
mask = median_filter(mask, size=3)  # 可选

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
plt.title("Text-Prior Attention + Logits Fusion (Hann window)")
plt.axis("off")
plt.show()