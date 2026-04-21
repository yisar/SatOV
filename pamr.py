import numpy as np
from PIL import Image
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels

def pamr_smooth_mask_fast(image_pil, mask_pil, num_iter=10, scale_factor=0.5):
    """
    通过缩放比例加速的 CRF 处理
    scale_factor: 缩放比例，建议 0.25 到 0.5
    """
    # 1. 记录原始尺寸
    orig_w, orig_h = image_pil.size
    
    # 2. 缩小图像（关键：大幅提升速度）
    target_size = (int(orig_w * scale_factor), int(orig_h * scale_factor))
    img_small = image_pil.resize(target_size, Image.BILINEAR)
    mask_small = mask_pil.resize(target_size, Image.NEAREST) # 掩码必须用最近邻插值
    
    img_np = np.array(img_small)
    mask_np = np.array(mask_small)
    H, W = img_np.shape[:2]

    # 3. 提取颜色映射（优化版）
    # 强制将 mask 重塑为 2D 颜色数组
    flat_mask = mask_np.reshape(-1, 3)
    unique_colors, labels = np.unique(flat_mask, axis=0, return_inverse=True)
    n_labels = len(unique_colors)
    label_mask = labels.reshape(H, W)

    print(f"检测到类别数: {n_labels} | 处理分辨率: {W}x{H}")

    # 4. CRF 核心逻辑
    d = dcrf.DenseCRF2D(W, H, n_labels)
    unary = unary_from_labels(label_mask.astype(np.int32), n_labels, gt_prob=0.9)
    d.setUnaryEnergy(unary)

    # 降低参数强度以适应缩小后的尺寸
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=30, srgb=13, rgbim=img_np, compat=10)

    Q = d.inference(num_iter)
    final_label = np.argmax(Q, axis=0).reshape(H, W)

    # 5. 还原颜色
    refined_mask_small = unique_colors[final_label].astype(np.uint8)
    
    # 6. 拉伸回原始尺寸
    return Image.fromarray(refined_mask_small).resize((orig_w, orig_h), Image.NEAREST)

if __name__ == "__main__":
    img = Image.open("./benchmark/UDD/origin/DJI_0303.JPG").convert("RGB")
    mask = Image.open("./benchmark/UDD/proxyclip/DJI_0303.JPG").convert("RGB")

    # scale_factor=0.3 表示只在 30% 的分辨率下计算 CRF
    smooth_mask_pil = pamr_smooth_mask_fast(img, mask, num_iter=5, scale_factor=0.3)
    smooth_mask_pil.save("mask_smoothed_fast.png")
    print("✅ 处理完成！")