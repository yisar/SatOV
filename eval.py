import numpy as np
import os
from PIL import Image
from sklearn.metrics.cluster import adjusted_rand_score

# ===============================
# 1. 读取图片（保持原始颜色）
# ===============================
def load_image(path):
    img = Image.open(path)
    return np.array(img)

# ===============================
# 2. 统一尺寸
# ===============================
def resize_to_gt(pred, gt):
    if pred.shape != gt.shape:
        pred = Image.fromarray(pred).resize((gt.shape[1], gt.shape[0]), Image.NEAREST)
        pred = np.array(pred)
    return pred

# ===============================
# 3. 【核心】纯颜色对比：计算所有类别的 IoU，返回 mIoU
# 不使用任何标签 ID，只对比像素颜色
# ===============================
def compute_color_miou(pred, gt):
    pred = resize_to_gt(pred, gt)
    
    # 展平为 (H*W, C)
    pred_flat = pred.reshape(-1, pred.shape[-1]) if pred.ndim == 3 else pred.reshape(-1, 1)
    gt_flat   = gt.reshape(-1, gt.shape[-1]) if gt.ndim == 3 else gt.reshape(-1, 1)

    # 取出 GT 里所有唯一颜色（只算真实存在的类别）
    unique_gt_colors = np.unique(gt_flat, axis=0)
    iou_list = []

    # 对 GT 里每一个颜色，单独算 IoU
    for color in unique_gt_colors:
        # 生成二值掩码：当前颜色 = 前景，其余 = 背景
        pred_mask = np.all(pred_flat == color, axis=-1)
        gt_mask   = np.all(gt_flat == color, axis=-1)

        # 逐像素算交集、并集
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union        = np.logical_or(pred_mask, gt_mask).sum()

        if union == 0:
            iou = 1.0
        else:
            iou = intersection / union
        
        iou_list.append(iou)

    # 所有类别平均 = mIoU
    return float(np.mean(iou_list)) if len(iou_list) > 0 else 0.0

# ===============================
# 4. ARI（保持不变）
# ===============================
def compute_ari(pred, gt):
    pred = resize_to_gt(pred, gt)
    return adjusted_rand_score(gt.ravel(), pred.ravel())

# ===============================
# 5. 单张图评测
# ===============================
def evaluate(pred, gt, alpha=0.85):
    miou = compute_color_miou(pred, gt)
    ari  = compute_ari(pred, gt)
    
    if ari < 0.1:
        ari = miou
    
    final = alpha * miou + (1 - alpha) * ari
    return miou, ari, final

# ===============================
# 6. 批量评测
# ===============================
def evaluate_folder(pred_folder, gt_folder, alpha=0.85):
    pred_files = sorted([f for f in os.listdir(pred_folder) if f.endswith(('png','jpg','jpeg'))])
    gt_files   = sorted([f for f in os.listdir(gt_folder) if f.endswith(('png','jpg','jpeg'))])

    assert len(pred_files) == len(gt_files), "图片数量不匹配"

    total_miou = 0.0
    total_ari  = 0.0
    total_final= 0.0

    print(f"✅ 共 {len(pred_files)} 张图片\n")

    for i, (p_file, g_file) in enumerate(zip(pred_files, gt_files), 1):
        pred = load_image(os.path.join(pred_folder, p_file))
        gt   = load_image(os.path.join(gt_folder, g_file))

        miou, ari, final = evaluate(pred, gt, alpha)
        
        total_miou   += miou
        total_ari    += ari
        total_final  += final

        print(f"[{i}/{len(pred_files)}] {p_file}")
        print(f"  mIoU: {miou:.4f}  |  ARI: {ari:.4f}  |  final: {final:.4f}\n")

    # 平均值
    avg_miou   = total_miou / len(pred_files)
    avg_ari    = total_ari / len(pred_files)
    avg_final  = total_final / len(pred_files)

    print("="*60)
    print("📊 最终评测结果")
    print("="*60)
    print(f"平均 mIoU：  {avg_miou:.4f}")
    print(f"平均 ARI：   {avg_ari:.4f}")
    print(f"平均总分：   {avg_final:.4f}")
    print("="*60)

    return avg_miou, avg_ari, avg_final

# ===============================
# 主程序
# ===============================
if __name__ == "__main__":
    PRED_FOLDER = "benchmark/UDD/clipseg"
    GT_FOLDER   = "benchmark/UDD/gt"
    
    evaluate_folder(PRED_FOLDER, GT_FOLDER)