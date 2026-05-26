import os
import matplotlib.pyplot as plt
from PIL import Image

# ====================== 请在这里修改你的根文件夹路径 ======================
ROOT_FOLDER = r"./benchmark"  # 例如：r"D:\data\models"
# ==========================================================================

# 设置中文字体（解决标题中文乱码）
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

# 收集所有图片路径（按顺序：3模型 × 7子文件夹 × 10张）
all_image_paths = []
all_image_titles = []

# 遍历 3 个模型文件夹
model_folders = sorted([f for f in os.listdir(ROOT_FOLDER)
                       if os.path.isdir(os.path.join(ROOT_FOLDER, f))])

for model in model_folders:
    model_path = os.path.join(ROOT_FOLDER, model)
    # 遍历每个模型下的 7 个子文件夹
    sub_folders = sorted([f for f in os.listdir(model_path)
                         if os.path.isdir(os.path.join(model_path, f))])

    for sub in sub_folders:
        sub_path = os.path.join(model_path, sub)
        # 获取该文件夹下的 10 张图片
        images = sorted([f for f in os.listdir(sub_path)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])

        for img in images:
            img_path = os.path.join(sub_path, img)
            all_image_paths.append(img_path)
            all_image_titles.append(img)  # 标题 = 文件名

# ====================== 绘图：每行10张，共21行 ======================
rows = 21
cols = 10
fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
axes = axes.flatten()  # 展平方便遍历

for idx, (img_path, title) in enumerate(zip(all_image_paths, all_image_titles)):
    if idx >= len(axes):
        break
    img = Image.open(img_path).convert("RGB")
    axes[idx].imshow(img)
    # axes[idx].set_title(title, fontsize=8)
    axes[idx].axis("off")  # 关闭坐标轴

# 隐藏多余子图
for idx in range(len(all_image_paths), len(axes)):
    axes[idx].axis("off")

plt.tight_layout()
# plt.suptitle("所有图片展示（3×7×10）", fontsize=16, y=0.98)
plt.subplots_adjust(top=0.96)
plt.show()