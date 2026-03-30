import numpy as np
import torch
import cv2
from PIL import Image
from transformers import SamModel, SamProcessor

# ===================== 你的配置 =====================
device = "cuda" if torch.cuda.is_available() else "cpu"

# 9 分类标签
target_labels = [
    'background', 'bareland', 'pavement', 'road', 'water',
    'tree', 'grass', 'cropland', 'building'
]

# 自定义调色板（一一对应）
custom_palette = [
    (68, 1, 84),   (72, 40, 120), (62, 74, 137),  (49, 104, 142),
    (38, 130, 142), (31, 158, 137), (73, 193, 110), (160, 218, 57),
    (253, 231, 37)
]

# 置信度
conf_threshold = 0.5

# ===================== 加载模型 =====================
print("Loading SAM model from transformers...")
model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

# ===================== 核心分割函数 =====================
def segment_with_labels(image_path, output_mask_path="mask.png"):
    # 1. 读取图片
    image = Image.open(image_path).convert("RGB")
    h, w = image.height, image.width

    # 2. 创建空的类别掩码
    final_mask = np.zeros((h, w), dtype=np.uint8)

    # 3. 遍历所有类别进行分割
    for class_id, label in enumerate(target_labels):
        if label == "background":
            continue  # 背景默认 0

        print(f"Processing: {label} (class {class_id})")

        # 全图分割（不需要点、不需要框）
        inputs = processor(
            images=image,
            input_points=[[[0, 0]]],  # 占位用，让模型执行全图分割
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # 后处理得到 mask
        mask = processor.post_process_masks(
            outputs.pred_masks,
            inputs["original_sizes"],
            inputs["reshaped_input_sizes"],
            binarize=True
        )[0][0][0].cpu().numpy()

        # 把当前类的区域写入最终掩码
        final_mask[mask > conf_threshold] = class_id

    # 4. 生成彩色掩码
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, color in enumerate(custom_palette):
        color_mask[final_mask == cid] = color

    # 5. 保存结果
    cv2.imwrite(output_mask_path, cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))
    print(f"\n✅ 掩码保存完成：{output_mask_path}")
    return final_mask, color_mask

# ===================== 运行 =====================
if __name__ == "__main__":
    INPUT_IMAGE = "img3.jpg"      # 你的图片路径
    OUTPUT_IMAGE = "result.png"   # 输出掩码路径
    segment_with_labels(INPUT_IMAGE, OUTPUT_IMAGE)