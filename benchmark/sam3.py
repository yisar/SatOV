import numpy as np
import torch
import cv2
from PIL import Image
from transformers import Sam3Model, Sam3Processor  # 👈 换成 SAM3

# ===================== 你的配置（完全不变） =====================
device = "cuda" if torch.cuda.is_available() else "cpu"

# 9 分类标签
target_labels = [
    'background', 'bareland', 'pavement', 'road', 'water',
    'tree', 'grass', 'cropland', 'building'
]

# 自定义调色板（完全不变）
custom_palette = [
    (68, 1, 84),   (72, 40, 120), (62, 74, 137),  (49, 104, 142),
    (38, 130, 142), (31, 158, 137), (73, 193, 110), (160, 218, 57),
    (253, 231, 37)
]

# 置信度
conf_threshold = 0.5

# ===================== 加载 SAM3 模型 =====================
print("Loading SAM3 model from transformers...")
model = Sam3Model.from_pretrained("facebook/sam3").to(device)
processor = Sam3Processor.from_pretrained("facebook/sam3")

# ===================== 核心分割函数（逻辑完全不变） =====================
def segment_with_labels(image_path, output_mask_path="mask.png"):
    # 1. 读取图片（完全不变）
    image = Image.open(image_path).convert("RGB")
    h, w = image.height, image.width

    # 2. 创建空的类别掩码（完全不变）
    final_mask = np.zeros((h, w), dtype=np.uint8)

    # 3. 遍历所有类别进行分割（内部换成 SAM3 文本分割）
    for class_id, label in enumerate(target_labels):
        if label == "background":
            continue  # 背景默认 0

        print(f"Processing: {label} (class {class_id})")

        # ===================== SAM3 文本分割核心 =====================
        # 不需要点！不需要框！直接输入文字！
        inputs = processor(
            images=image,
            text=label,  # 👈 直接用类别名称做文本提示
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # SAM3 专用后处理（实例分割 + 置信度过滤）
        result = processor.post_process_instance_segmentation(
            outputs,
            threshold=conf_threshold,
            target_sizes=[[h, w]]  # 缩放到原图尺寸
        )[0]

        # 获取 mask
        mask = np.zeros((h, w), dtype=bool)
        if "masks" in result and len(result["masks"]) > 0:
            masks_np = result["masks"].cpu().numpy()
            mask = np.any(masks_np, axis=0)  # 合并所有该类掩码

        # 把当前类的区域写入最终掩码（完全不变）
        final_mask[mask] = class_id

    # 4. 生成彩色掩码（完全不变）
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, color in enumerate(custom_palette):
        color_mask[final_mask == cid] = color

    # 5. 保存结果（完全不变）
    cv2.imwrite(output_mask_path, cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))
    print(f"\n✅ 掩码保存完成：{output_mask_path}")
    return final_mask, color_mask

# ===================== 运行（完全不变） =====================
if __name__ == "__main__":
    INPUT_IMAGE = "img3.jpg"      # 你的图片路径
    OUTPUT_IMAGE = "result.png"   # 输出掩码路径
    segment_with_labels(INPUT_IMAGE, OUTPUT_IMAGE)