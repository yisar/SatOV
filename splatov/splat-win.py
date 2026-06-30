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
from satup.gsup import GaussianFeatureUpsampler, create_coordinate_grid_2d

# =========================
# 1. Load model
# =========================
model, tokenizer = dinov3_vitl16_dinotxt_tet1280d20h24l()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device).eval()

# =========================
# 2. Input image and class labels
# =========================
image_path = "asset/img.jpg"
image = Image.open(image_path).convert("RGB")
orig_w, orig_h = image.size
H_img, W_img = orig_h, orig_w

class_names = [
    # "background",
    "pavement",
    "road",
    "forest",
    "grass",
    "field",
    # "cropland"
    "river",
    "building",
    "hourse",
]
texts = [f"a photo of {c}" for c in class_names]
num_classes = len(class_names)

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
# 5. Tokenize text globally
# =========================
text_tokens = tokenizer.tokenize(texts).to(device)

# =========================
# 6. Accumulators on GPU
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
        h = w = int(P**0.5)  # patch grid size
        img_feat = image_patch_tokens.transpose(1, 2).reshape(B, D, h, w)
        text_feat = model.encode_text(text_tokens)[:, 1024:]
        img_feat = F.normalize(img_feat, dim=1)
        text_feat = F.normalize(text_feat, dim=-1)
        logits = torch.einsum("bchw,nc->bnhw", img_feat, text_feat)  # [1, C, h, w]
        prob = torch.softmax(logits / 0.07, dim=1)  # [1, C, h, w]

        # Gaussian JBU upsampling to win_size
        prob_lr = rearrange(prob[0], "c h w -> (h w) c").unsqueeze(0)  # [1, h*w, C]
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
        upsampled = upsampler.forward(prob_lr)  # [1, win_size*win_size, C]
        up_prob = upsampled.reshape(1, win_size, win_size, num_classes).permute(
            0, 3, 1, 2
        )  # [1, C, 224, 224]
        up_prob = up_prob.squeeze(0)  # [C, 224, 224]

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
final_logits = acc_logits / acc_weights  # [C, H, W]
final_logits = final_logits.unsqueeze(0)  # [1, C, H, W]

# Depthwise Gaussian smoothing kernel (3x3)
kernel = (
    torch.tensor(
        [[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32, device=device
    ).view(1, 1, 3, 3)
    / 16
)
kernel = kernel.repeat(num_classes, 1, 1, 1)  # [C, 1, 3, 3]
final_logits_smooth = F.conv2d(
    final_logits, weight=kernel, padding=1, groups=num_classes
)

# Convert to numpy and argmax
final_logits_np = final_logits_smooth.squeeze(0).cpu().numpy()  # [C, H, W]
mask = np.argmax(final_logits_np, axis=0)  # [H, W]

# Optional median filter to remove small isolated blobs
mask = median_filter(mask, size=3)

# =========================
# 9. Visualization
# =========================
custom_palette = [
    # (68, 1, 84),
    (72, 40, 120),
    (62, 74, 137),
    (49, 104, 142),
    (38, 130, 142),
    (31, 158, 137),
    (73, 193, 110),
    (160, 218, 57),
    (253, 231, 37),
]
custom_palette = np.array(custom_palette) / 255.0
cmap = ListedColormap(custom_palette)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Input")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(mask, cmap=cmap, vmin=0, vmax=num_classes - 1)
plt.title("Sliding Window + Gaussian JBU")
plt.axis("off")

plt.show()
