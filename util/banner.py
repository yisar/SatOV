import os
import matplotlib.pyplot as plt
from PIL import Image

# ========== 按需修改参数 ==========
FOLDER_LIST = [r"./bench/banner/UDD", r"./bench/banner/SSSI", r"./bench/banner/DDOA"]
BASE_NAMES = ["origin", "maskclip", "clipseg", "clearclip", "lposs", "ours"]
IMAGE_HEIGHT = 300  # 图片统一像素高度
DPI = 150
IMG_EXTS = (".jpg", ".png", ".jpeg")
SAVE_PATH = "多文件夹多行拼接结果.jpg"

# 子图间距
WSPACE = 0.01
HSPACE = 0.005
# 整体边距
LEFT = 0.01
RIGHT = 0.99
TOP = 0.94
BOTTOM = 0.01
# =================================

all_rows_imgs = []
col_titles = BASE_NAMES

# 读取并缩放所有图片
for folder in FOLDER_LIST:
    row_imgs = []
    for base_name in BASE_NAMES:
        img_path = None
        for ext in IMG_EXTS:
            temp_path = os.path.join(folder, base_name + ext)
            if os.path.exists(temp_path):
                img_path = temp_path
                break
        if img_path is None:
            print(f"⚠️ 文件夹 {folder} 缺失文件: {base_name}")
            continue

        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        new_w = int(w * IMAGE_HEIGHT / h)
        img = img.resize((new_w, IMAGE_HEIGHT), Image.Resampling.LANCZOS)
        row_imgs.append(img)
    all_rows_imgs.append(row_imgs)

row_num = len(all_rows_imgs)
col_num = len(BASE_NAMES)

# 动态计算画布尺寸：像素转英寸
inch_h = IMAGE_HEIGHT / DPI
# 总画布宽高
fig_width = col_num * inch_h * 1.3
fig_height = row_num * inch_h + 0.4

fig = plt.figure(figsize=(fig_width, fig_height))
plt.subplots_adjust(
    left=LEFT, right=RIGHT, top=TOP, bottom=BOTTOM, wspace=WSPACE, hspace=HSPACE
)

# 绘制图片
for row_idx, row_imgs in enumerate(all_rows_imgs):
    for col_idx, img in enumerate(row_imgs):
        ax = fig.add_subplot(row_num, col_num, row_idx * col_num + col_idx + 1)
        ax.imshow(img)
        if row_idx == 0:
            ax.set_title(col_titles[col_idx], fontsize=11, pad=3)
        ax.axis("off")

plt.savefig(SAVE_PATH, dpi=DPI)
plt.show()
print(f"✅ 图片已保存至: {SAVE_PATH}")
