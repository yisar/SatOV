import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import os
os.environ["GITHUB_TOKEN"] = "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"
# ------------------------------------------------------------
# Device
# ------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------
# Load Backbone
# (这里以 DINOv2 为例，可替换为任意 VFM)
# ------------------------------------------------------------
backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").to(device).eval()

# ------------------------------------------------------------
# Load NAF
# ------------------------------------------------------------
naf = torch.hub.load(
    "valeoai/NAF",
    "naf",
    pretrained=True,
    device=device,
).to(device)

ckpt = torch.load("naf.pth", map_location=device)
naf.load_state_dict(ckpt)
naf.eval()

# ------------------------------------------------------------
# Image
# ------------------------------------------------------------
image = Image.open("demo.jpg").convert("RGB")

transform = T.Compose(
    [
        T.Resize((448, 448)),
        T.ToTensor(),
    ]
)

img = transform(image).unsqueeze(0).to(device)

# ------------------------------------------------------------
# Different normalization
# ------------------------------------------------------------
# DINOv2 normalization
img_backbone = T.functional.normalize(
    img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
)

# NAF normalization（官方也是 ImageNet）
img_naf = T.functional.normalize(
    img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
)

# ------------------------------------------------------------
# Extract feature
# ------------------------------------------------------------
with torch.no_grad():
    tokens = backbone.forward_features(img_backbone)

    feat = tokens["x_norm_patchtokens"]

    B, N, C = feat.shape

    H = W = int(N**0.5)

    feat = feat.transpose(1, 2).reshape(B, C, H, W)

print("Input feature:", feat.shape)

# ------------------------------------------------------------
# Upsample
# ------------------------------------------------------------
with torch.no_grad():
    pred448 = naf(img_naf, feat, (448, 448))

print(pred448.shape)

# ------------------------------------------------------------
# Visualize
# ------------------------------------------------------------
vis = pred448.norm(dim=1)

vis = (vis - vis.min()) / (vis.max() - vis.min())

plt.figure(figsize=(6, 6))
plt.imshow(vis[0].cpu(), cmap="viridis")
plt.axis("off")
plt.show()
