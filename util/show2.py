import os
import matplotlib.pyplot as plt
from PIL import Image

# =========================
# 配置
# =========================
folder_info = [
    (r"./bench/UDD/ours", "UDD"),
    (r"./bench/DDOA/ours", "DDOA"),
    (r"./bench/SSSI/ours", "SSSI"),
]

TARGET_WIDTH = 256
IMG_COLS = 5

# =========================
# 版式参数
# =========================
LEFT_MARGIN = 0.001
RIGHT_MARGIN = 0.001

IMG_GAP_X = 0.002
ROW_GAP = 0.0025
GROUP_GAP = 0.005

TITLE_FONT_SIZE = 14  # 🔥 button size

# =========================
# 中文支持
# =========================
plt.rcParams["font.family"] = [
    "SimHei",
    "WenQuanYi Micro Hei",
    "Heiti TC",
]
plt.rcParams["axes.unicode_minus"] = False

# =========================
# 读取图片
# =========================
all_groups = []

for path, title in folder_info:
    img_files = sorted(
        [
            f
            for f in os.listdir(path)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ]
    )

    imgs = []

    for fname in img_files:
        img = Image.open(os.path.join(path, fname)).convert("RGB")

        w, h = img.size
        scale = TARGET_WIDTH / w

        new_w = TARGET_WIDTH
        new_h = int(h * scale)

        img = img.resize(
            (new_w, new_h),
            Image.Resampling.LANCZOS,
        )

        imgs.append(
            {
                "img": img,
                "ratio": new_h / new_w,
            }
        )

    all_groups.append((title, imgs))

# =========================
# 固定列宽
# =========================
img_w = (1 - LEFT_MARGIN - RIGHT_MARGIN - IMG_GAP_X * (IMG_COLS - 1)) / IMG_COLS

# =========================
# 计算总高度
# =========================
group_heights = []

for _, imgs in all_groups:
    row1 = imgs[:IMG_COLS]
    row2 = imgs[IMG_COLS : IMG_COLS * 2]

    row1_h = max(img_w * i["ratio"] for i in row1) if row1 else 0
    row2_h = max(img_w * i["ratio"] for i in row2) if row2 else 0

    group_heights.append(row1_h + ROW_GAP + row2_h)

total_height = sum(group_heights) + GROUP_GAP * (len(group_heights) - 1)

# =========================
# 创建画布
# =========================
fig = plt.figure(
    figsize=(14, total_height * 12),
    dpi=300,
)

# =========================
# 绘制
# =========================
y_cursor = total_height

for group_idx, (title, imgs) in enumerate(all_groups):
    row1 = imgs[:IMG_COLS]
    row2 = imgs[IMG_COLS : IMG_COLS * 2]

    row1_h = max(img_w * i["ratio"] for i in row1) if row1 else 0
    row2_h = max(img_w * i["ratio"] for i in row2) if row2 else 0

    # =====================
    # 第一行
    # =====================
    y_cursor -= row1_h

    for col, item in enumerate(row1):
        h = img_w * item["ratio"]

        x = LEFT_MARGIN + col * (img_w + IMG_GAP_X)

        y = (y_cursor + (row1_h - h) / 2) / total_height

        ax = fig.add_axes(
            [
                x,
                y,
                img_w,
                h / total_height,
            ]
        )

        ax.imshow(item["img"])
        ax.axis("off")

    # =====================
    # 行间距
    # =====================
    y_cursor -= ROW_GAP

    # =====================
    # 第二行
    # =====================
    y_cursor -= row2_h

    for col, item in enumerate(row2):
        h = img_w * item["ratio"]

        x = LEFT_MARGIN + col * (img_w + IMG_GAP_X)

        y = (y_cursor + (row2_h - h) / 2) / total_height

        ax = fig.add_axes(
            [
                x,
                y,
                img_w,
                h / total_height,
            ]
        )

        ax.imshow(item["img"])
        ax.axis("off")

    # =====================
    # 🔥 左上角 BUTTON TAG（修复版）
    # =====================

    row1_top = y_cursor + row1_h + ROW_GAP + row2_h

    tag_x = LEFT_MARGIN + 0.005
    tag_y = (row1_top - 0.003) / total_height

    fig.text(
        tag_x+0.04,
        tag_y-0.03,
        title,
        ha="left",
        va="top",
        fontsize=28,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.4",  # 🔥 更像 button
            facecolor=(1, 1, 1, 0.9),
            edgecolor="black",
            linewidth=1.2,
        ),
    )

    # =====================
    # group gap
    # =====================
    if group_idx < len(all_groups) - 1:
        y_cursor -= GROUP_GAP

# =========================
# 保存
# =========================
save_path = "./result.png"

plt.savefig(
    save_path,
    dpi=300,
    bbox_inches="tight",
    pad_inches=0,
)

plt.close()

print(f"已保存至: {save_path}")
