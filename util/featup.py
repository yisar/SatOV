import torch
from PIL import Image
import matplotlib.pyplot as plt
import os

# 1. 设备配置
device = "cuda" if torch.cuda.is_available() else "cpu"

os.environ["GIT_HTTPS_PROXY"] = "https://ghproxy.net/https://github.com"
os.environ["TORCH_HUB_HTTPS_PROXY"] = "https://ghproxy.net"

device = "cuda" if torch.cuda.is_available() else "cpu"
# 只加载backbone特征提取器，不加载带cuda算子的上采样头
feat_extractor = torch.hub.load(
    "mhamilton723/FeatUp",
    "dino16",
    pretrained=True,
    device=device,
    trust_repo=True
)

img = Image.open("test.jpg").convert("RGB")
low_feat = feat_extractor(img)
# 原生torch上采样，无需任何自定义算子、无需编译
high_res_feat = torch.nn.functional.interpolate(low_feat, scale_factor=2, mode="bilinear")
print(high_res_feat.shape)



print(f"输入图像尺寸: {img.size}")

# 5. 可视化特征通道示例（取第0个通道）
feat_map = high_res_feat[0, 0].detach().cpu().numpy()

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Input Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(feat_map, cmap="viridis")
plt.title("FeatUp Upsampled Feature Map")
plt.axis("off")

plt.tight_layout()
plt.show()
