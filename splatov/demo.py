from pathlib import Path
import sys

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))
from dinov3.data.transforms import make_classification_eval_transform
from dinov3.hub.dinotxt import dinov3_vitl16_dinotxt_tet1280d20h24l

# =========================
# 1. load dinov3-txt model (CORRECT WAY)
# =========================


model, tokenizer = dinov3_vitl16_dinotxt_tet1280d20h24l()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device).eval()

# =========================
# 2. input image + classes
# =========================
image_path = "asset/256.png"   # 改成你的图片路径
image = Image.open(image_path).convert("RGB")

class_names = ["road", "building","grass","house","tree" "river", "background"]
texts = [f"a photo of {c}" for c in class_names]

# =========================
# 3. preprocess image (IMPORTANT)
# =========================
transform = make_classification_eval_transform()
image_tensor = torch.stack([transform(image)], dim=0).to(device)

# =========================
# 4. tokenize text (IMPORTANT)
# =========================
text_tokens = tokenizer.tokenize(texts).to(device)

# =========================
# 5. forward pass
# =========================
with torch.no_grad():
    # image patch features
    image_cls, image_patch_tokens, _ = model.encode_image_with_patch_tokens(image_tensor)

    # text features
    text_features = model.encode_text(text_tokens)

# =========================
# 6. reshape patch tokens -> feature map
# =========================
B, P, D = image_patch_tokens.shape
H = W = int(P ** 0.5)

img_feat = image_patch_tokens.transpose(1, 2).reshape(B, D, H, W)

# =========================
# 7. text aligned features
# =========================
text_feat = text_features[:, 1024:]  # dinotxt aligned part

# =========================
# 8. normalize
# =========================
img_feat = F.normalize(img_feat, dim=1)
text_feat = F.normalize(text_feat, dim=-1)

# =========================
# 9. pixel-wise similarity
# =========================
logits = torch.einsum("bchw,nc->bnhw", img_feat, text_feat)

mask = logits.argmax(dim=1)[0]  # [H, W]

# =========================
# 10. upscale to original image size
# =========================
mask = F.interpolate(
    mask.unsqueeze(0).unsqueeze(0).float(),
    size=image.size[::-1],
    mode="nearest"
)[0, 0].cpu().numpy()

# =========================
# 11. visualization
# =========================
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Input Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(mask, cmap="jet")
plt.title("Zero-shot Segmentation (DINOv3-TXT)")
plt.axis("off")

plt.show()