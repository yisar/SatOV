import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# ==========================================================
# Configuration
# ==========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

image_path = "asset/9.png"      # 修改成你的图片

# ==========================================================
# Load DINOv2
# ==========================================================
model = torch.hub.load(
    "facebookresearch/dinov2",
    "dinov2_vitb14"
)

model.eval()
model.to(device)

# ==========================================================
# Image preprocessing
# ==========================================================
transform = transforms.Compose([
    transforms.Resize(518),
    transforms.CenterCrop(518),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])

img = Image.open(image_path).convert("RGB")
x = transform(img).unsqueeze(0).to(device)

# ==========================================================
# Forward
# ==========================================================
with torch.no_grad():
    features = model.forward_features(x)

# Patch Tokens
tokens = features["x_norm_patchtokens"][0]      # [1369,768]

num_patch = tokens.shape[0]
embed_dim = tokens.shape[1]

H = W = int(np.sqrt(num_patch))

print("Patch:", H, W)
print("Embedding:", embed_dim)

# ==========================================================
# Random choose channels
# ==========================================================
np.random.seed(0)

channels = np.random.choice(embed_dim, 12, replace=False)

fig, axes = plt.subplots(3,4, figsize=(10,8))

for ax, channel in zip(axes.flat, channels):

    feat = tokens[:, channel]

    feat = feat.reshape(H, W)

    # bicubic upsample
    feat = F.interpolate(
        feat[None,None],
        size=(518,518),
        mode="bicubic",
        align_corners=False
    )[0,0]

    feat = feat.cpu().numpy()

    # ----------------------------
    # z-score normalization
    # ----------------------------
    feat = (feat - feat.mean()) / (feat.std() + 1e-6)

    # clip
    feat = np.clip(feat, -2.5, 2.5)

    ax.imshow(
        feat,
        cmap="coolwarm",
        vmin=-2.5,
        vmax=2.5,
        interpolation="bicubic"
    )

    ax.set_title(f"Channel {channel}", fontsize=9)
    ax.axis("off")

plt.tight_layout()
plt.show()