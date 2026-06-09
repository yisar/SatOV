import os
import matplotlib.pyplot as plt
from PIL import Image

# ========== 按需修改参数 ==========
IMAGE_FOLDER = r"./bench/banner"
# 按指定顺序填写完整文件名(带后缀)
ORDERED_NAMES = ["origin.jpg", "maskclip.png", "clipseg.png", "clearclip.png", "lposs.png", "ours.png"]
IMAGE_HEIGHT = 300  # 图片统一高度
# =================================

# 加载图片+提取纯文件名
img_list = []
name_list = []
for file_name in ORDERED_NAMES:
    img_path = os.path.join(IMAGE_FOLDER, file_name)
    if not os.path.exists(img_path):
        print(f"⚠️ 缺失文件: {file_name}")
        continue

    # 读取并等比缩放图片
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    new_w = int(w * IMAGE_HEIGHT / h)
    img = img.resize((new_w, IMAGE_HEIGHT), Image.Resampling.LANCZOS)
    img_list.append(img)

    # 去除后缀，得到纯名称
    pure_name = os.path.splitext(file_name)[0]
    name_list.append(pure_name)

# 创建画布：一行多列子图
n = len(img_list)
plt.figure(figsize=(3 * n, 5))

for idx in range(n):
    plt.subplot(1, n, idx + 1)
    plt.imshow(img_list[idx])
    plt.title(name_list[idx], fontsize=12, pad=10)  # 文字在图片下方
    plt.axis("off")  # 隐藏坐标轴

plt.tight_layout()

# 👇 必须先保存，再 show！
plt.savefig("横向带名称长图.jpg", dpi=150, bbox_inches="tight")
plt.show()  # 保存完再显示