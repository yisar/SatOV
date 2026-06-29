from pathlib import Path
import sys

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.colors import ListedColormap   # 新增导入

root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))
from dinov3.data.transforms import make_classification_eval_transform
from dinov3.hub.dinotxt import dinov3_vitl16_dinotxt_tet1280d20h24l

# =========================
# 1. 加载模型（DINOv3‑TXT + AnyUp）
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

# DINOv3‑TXT 模型
model, tokenizer = dinov3_vitl16_dinotxt_tet1280d20h24l()
model = model.to(device).eval()

# AnyUp 引导上采样器
anyup = torch.hub.load("wimmerth/anyup", "anyup", verbose=False).to(device).eval()

# =========================
# 2. 输入图片 + 类别文本
# =========================
image_path = "asset/256.png"   # 请替换为实际路径
image = Image.open(image_path).convert("RGB")

class_names = ["road", "building", "grass", "house", "tree", "river", "background"]
texts = [f"a photo of {c}" for c in class_names]

# =========================
# 3. 预处理图像（DINOv3 标准变换）
# =========================
transform = make_classification_eval_transform()
image_tensor = torch.stack([transform(image)], dim=0).to(device)

# =========================
# 4. 文本分词
# =========================
text_tokens = tokenizer.tokenize(texts).to(device)

# =========================
# 5. 模型前向推理
# =========================
with torch.no_grad():
    # 图像 patch 特征
    image_cls, image_patch_tokens, _ = model.encode_image_with_patch_tokens(image_tensor)
    # 文本特征
    text_features = model.encode_text(text_tokens)

# =========================
# 6. 将 patch 序列重排为 2D 特征图
# =========================
B, P, D = image_patch_tokens.shape
H = W = int(P ** 0.5)
img_feat = image_patch_tokens.transpose(1, 2).reshape(B, D, H, W)

# =========================
# 7. 提取 DINOv3‑TXT 对齐的文本部分（后 1024 维）
# =========================
text_feat = text_features[:, 1024:]

# =========================
# 8. 特征归一化
# =========================
img_feat = F.normalize(img_feat, dim=1)
text_feat = F.normalize(text_feat, dim=-1)

# =========================
# 9. 计算像素级相似度 logits
# =========================
logits = torch.einsum("bchw,nc->bnhw", img_feat, text_feat)   # [B, N, H, W]

# =========================
# 10. 使用 AnyUp 引导上采样到原始图像尺寸
# =========================
target_height, target_width = image.size[::-1]   # (H_orig, W_orig)

# 准备高分辨率 RGB 引导图（需与最终输出尺寸一致）
rgb_hr = F.interpolate(
    image_tensor,
    size=(target_height, target_width),
    mode='bilinear',
    align_corners=False
)   # [B, 3, H_orig, W_orig]

# AnyUp 上采样：输入引导图 RGB 和低分辨率特征图 logits
with torch.no_grad():
    logits_upsampled = anyup(rgb_hr, logits)   # 输出形状 [B, N, H_orig, W_orig]

# 由 logits 取 argmax 得到硬分割掩码
mask = logits_upsampled.argmax(dim=1)[0].cpu().numpy()   # [H_orig, W_orig]

# =========================
# 11. 可视化结果（使用自定义颜色表）
# =========================
# 定义您提供的 9 种颜色（归一化到 0~1）
color_list = [
    (68/255, 1/255, 84/255),
    (72/255, 40/255, 120/255),
    (62/255, 74/255, 137/255),
    (49/255, 104/255, 142/255),
    (38/255, 130/255, 142/255),
    (31/255, 158/255, 137/255),
    (73/255, 193/255, 110/255),
    (160/255, 218/255, 57/255),
    (253/255, 231/255, 37/255),
]
# 取前 N 个（N = 类别数）
colors = color_list[:len(class_names)]
cmap = ListedColormap(colors)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Input Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(mask, cmap=cmap, vmin=0, vmax=len(colors)-1)
plt.title("Zero-shot Segmentation (DINOv3-TXT + AnyUp)")
plt.axis("off")

plt.show()