from pathlib import Path
import sys
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from einops import rearrange
import numpy as np
from matplotlib.colors import ListedColormap

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
# 1. model
# =========================
model, tokenizer = dinov3_vitl16_dinotxt_tet1280d20h24l()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device).eval()

# =========================
# 2. input
# =========================
image_path = "asset/256.png"
image = Image.open(image_path).convert("RGB")

# class_names = ["road", "building", "tree","grass","house" "river","house", "background"]
class_names = ["road", "building","grass","house","tree" "river", "background"]
texts = [f"a photo of {c}" for c in class_names]

transform = make_classification_eval_transform()
image_tensor = torch.stack([transform(image)], dim=0).to(device)

# =========================
# 3. forward
# =========================
text_tokens = tokenizer.tokenize(texts).to(device)

with torch.no_grad():
    _, image_patch_tokens, _ = model.encode_image_with_patch_tokens(image_tensor)
    text_features = model.encode_text(text_tokens)

# =========================
# 4. patch reshape
# =========================
B, P, D = image_patch_tokens.shape
H = W = int(P ** 0.5)

img_feat = image_patch_tokens.transpose(1, 2).reshape(B, D, H, W)

# =========================
# 5. text features
# =========================
text_feat = text_features[:, 1024:]

img_feat = F.normalize(img_feat, dim=1)
text_feat = F.normalize(text_feat, dim=-1)

# =========================
# 6. logits
# =========================
logits = torch.einsum("bchw,nc->bnhw", img_feat, text_feat)

prob = torch.softmax(logits / 0.07, dim=1)

# =========================
# 7. Gaussian JBU (RGB guided)
# =========================
with torch.no_grad():
    B, C, h, w = prob.shape
    H_img, W_img = image_tensor.shape[-2:]

    prob_lr = rearrange(prob[0], "c h w -> (h w) c").unsqueeze(0)

    patch_coords_lr = create_coordinate_grid_2d(h, w, device).reshape(-1, 2).unsqueeze(0)
    patch_coords_hr = create_coordinate_grid_2d(H_img, W_img, device).reshape(-1, 2).unsqueeze(0)

    pixels_lr = rearrange(image_tensor, "b c h w -> b (h w) c")
    pixels_hr = pixels_lr

    feature_upsampler = GaussianFeatureUpsampler(
        patch_coords_lr=patch_coords_lr,
        patch_coords_hr=patch_coords_hr,
        pixels_lr=pixels_lr,
        pixels_hr=pixels_hr,
    ).to(device)

# =========================
# 8. run JBU
# =========================
with torch.no_grad():
    upsampled = feature_upsampler.forward(prob_lr)

# =========================
# 9. FIXED reshape (核心修复点)
# =========================

B = 1
C = upsampled.shape[-1]

up_hr = upsampled.reshape(B, H_img, W_img, C).permute(0, 3, 1, 2)

# =========================
# 10. mask
# =========================
mask = up_hr.argmax(dim=1)[0].cpu().numpy()

# =========================
# 11. visualization
# =========================
custom_palette = [
    (68, 1, 84),
    (49, 104, 142),
    (38, 130, 142),
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

mask_vis = mask % len(custom_palette)

plt.imshow(mask_vis, cmap=cmap)
plt.title("Fixed Gaussian JBU")
plt.axis("off")

plt.show()