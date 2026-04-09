import numpy as np
from PIL import Image
import random

# ===============================
# 加载 mask（你原来的）
# ===============================
def load_mask(path):
    img = Image.open(path)
    if img.mode == "RGB":
        return rgb_to_label(img)
    else:
        return np.array(img)

def rgb_to_label(img):
    img = np.array(img)
    h, w, _ = img.shape
    label = np.zeros((h, w), dtype=np.int32)
    colors = np.unique(img.reshape(-1, 3), axis=0)
    for idx, color in enumerate(colors):
        mask = np.all(img == color, axis=-1)
        label[mask] = idx
    return label

# ===============================
# 🔥 双像素对比法（真正靠谱无监督分数）
# ===============================
def pixel_pair_score(img_rgb, mask, n_samples=20000):
    img = np.array(img_rgb) / 255.0  # 归一化
    h, w = mask.shape

    same_dist = []  # 同类像素的差距
    diff_dist = []  # 异类像素的差距

    # 随机采样 20000 对像素对比（速度快）
    for _ in range(n_samples):
        x1, y1 = random.randint(0, h-1), random.randint(0, w-1)
        x2, y2 = random.randint(0, h-1), random.randint(0, w-1)

        l1 = mask[x1, y1]
        l2 = mask[x2, y2]

        # 计算两个像素的颜色差
        p1 = img[x1, y1]
        p2 = img[x2, y2]
        dist = np.linalg.norm(p1 - p2)  # 欧式距离

        if l1 == l2:
            same_dist.append(dist)
        else:
            diff_dist.append(dist)

    # 平均差距
    same = np.mean(same_dist) if same_dist else 1.0
    diff = np.mean(diff_dist) if diff_dist else 0.0

    # 最终无监督分数（越高越好）
    quality = diff - same
    norm_score = 1.0 / (1.0 + np.exp(-quality * 8))  # 归一化到 0~1

    return {
        "同类内部平均差距（越小越好）": round(same, 4),
        "异类之间平均差距（越大越好）": round(diff, 4),
        "无监督分割质量分数（0~1）": round(norm_score, 4)
    }

# ===============================
# 主程序
# ===============================
if __name__ == "__main__":
    # 你的路径
    img_path = "dataset/DDOA/P1435.png"
    mask_path = "res/DDOA/P1435.png"

    img = Image.open(img_path).convert("RGB")
    mask = load_mask(mask_path)

    # 🔥 计算分数
    score = pixel_pair_score(img, mask)
    print("\n=== 双像素对比法 无监督分割分数 ===")
    for k, v in score.items():
        print(f"{k}: {v}")