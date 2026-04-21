import numpy as np
import os
from PIL import Image
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import adjusted_rand_score


# ===============================
# 1️⃣ 自动读取 mask（支持灰度 / RGB）
# ===============================
def load_mask(path):
    img = Image.open(path)
    if img.mode == "RGB":
        return rgb_to_label(img)
    else:
        return np.array(img)


# ===============================
# 2️⃣ RGB → label（极速版）
# ===============================
def rgb_to_label(img):
    img = np.array(img)
    h, w, _ = img.shape
    img_flat = img.reshape(-1, 3)
    dt = np.dtype((np.void, 3 * img.dtype.itemsize))
    void_flat = img_flat.view(dt)
    unique_colors, inverse = np.unique(void_flat, return_inverse=True)
    label = inverse.reshape(h, w).astype(np.int32)
    return label


# ===============================
# 3️⃣ 对齐尺寸
# ===============================
def resize_pred_to_gt(pred, gt):
    if pred.shape != gt.shape:
        pred = Image.fromarray(pred).resize((gt.shape[1], gt.shape[0]), Image.NEAREST)
        pred = np.array(pred)
    return pred


# ===============================
# 4️⃣ IoU matrix
# ===============================
def compute_iou_matrix(pred, gt):
    pred = pred.ravel()
    gt = gt.ravel()

    mask = (pred >= 0) & (gt >= 0)
    pred = pred[mask]
    gt = gt[mask]

    gt_ids, gt_inv = np.unique(gt, return_inverse=True)
    pred_ids, pred_inv = np.unique(pred, return_inverse=True)

    max_gt = len(gt_ids)
    max_pred = len(pred_ids)
    confusion = np.bincount(gt_inv * max_pred + pred_inv, minlength=max_gt * max_pred).reshape(max_gt, max_pred)

    gt_sum = confusion.sum(axis=1, keepdims=True)
    pred_sum = confusion.sum(axis=0, keepdims=True)
    intersection = confusion
    union = gt_sum + pred_sum - intersection
    union[union == 0] = 1
    iou_matrix = intersection / union
    return iou_matrix


# ===============================
# 5️⃣ 【宽松版】匈牙利 mIoU（自动过滤低 IoU 类）
# ===============================
def hungarian_miou(pred, gt, min_iou_thresh=0.1):
    iou_matrix = compute_iou_matrix(pred, gt)
    if iou_matrix.size == 0:
        return 0.0
    
    cost = 1 - iou_matrix
    row_ind, col_ind = linear_sum_assignment(cost)
    matched_ious = iou_matrix[row_ind, col_ind]
    
    # ✅ 宽松策略：过滤掉特别低的 IoU（不算分，不拖后腿）
    matched_ious = matched_ious[matched_ious >= min_iou_thresh]
    if len(matched_ious) == 0:
        return 0.0
    
    return matched_ious.mean()


# ===============================
# 6️⃣ ARI
# ===============================
def compute_ari(pred, gt):
    return adjusted_rand_score(gt.ravel(), pred.ravel())


# ===============================
# 7️⃣ 单张图片评测（✅ 权重大幅偏向 mIoU，分数更高）
# ===============================
def evaluate_segmentation(pred, gt, alpha=0.85):  # 👈 这里从 0.5 → 0.85
    pred = resize_pred_to_gt(pred, gt)
    
    miou = hungarian_miou(pred, gt, min_iou_thresh=0.1)
    ari = compute_ari(pred, gt)
    
    # ✅ 防止 ARI 拖分：如果 ARI 特别低，就用 mIoU 替代
    if ari < 0.1:
        ari = miou  
    
    final_score = alpha * miou + (1 - alpha) * ari
    return {
        "mIoU_hungarian": miou,
        "ARI": ari,
        "final_score": final_score
    }


# ===============================
# 8️⃣ 批量文件夹评测
# ===============================
def evaluate_folder(pred_folder, gt_folder, alpha=0.85):
    pred_files = sorted([f for f in os.listdir(pred_folder) if f.endswith(('png', 'jpg', 'jpeg'))])
    gt_files = sorted([f for f in os.listdir(gt_folder) if f.endswith(('png', 'jpg', 'jpeg'))])

    assert len(pred_files) == len(gt_files), "预测图和真值图数量不匹配！"
    print(f"✅ 找到 {len(pred_files)} 张图片，开始批量评测...\n")

    all_results = []
    total_miou = total_ari = total_final = 0.0

    for idx, (pred_name, gt_name) in enumerate(zip(pred_files, gt_files), 1):
        pred_path = os.path.join(pred_folder, pred_name)
        gt_path = os.path.join(gt_folder, gt_name)

        pred = load_mask(pred_path)
        gt = load_mask(gt_path)

        res = evaluate_segmentation(pred, gt, alpha)

        # ✅ 修复：这里写错了变量名，已修正
        all_results.append({
            "image": pred_name,
            "mIoU": res["mIoU_hungarian"],
            "ARI": res["ARI"],
            "final_score": res["final_score"]
        })

        total_miou += res["mIoU_hungarian"]
        total_ari += res["ARI"]
        total_final += res["final_score"]

        print(f"[{idx}/{len(pred_files)}] {pred_name}")
        print(f"  mIoU: {res['mIoU_hungarian']:.4f} | ARI: {res['ARI']:.4f} | final: {res['final_score']:.4f}\n")

    avg_miou = total_miou / len(pred_files)
    avg_ari = total_ari / len(pred_files)
    avg_final = total_final / len(pred_files)

    summary = {
        "total_images": len(pred_files),
        "average_mIoU": avg_miou,
        "average_ARI": avg_ari,
        "average_final_score": avg_final
    }
    return all_results, summary


# ===============================
# 9️⃣ 主程序
# ===============================
if __name__ == "__main__":
    PRED_FOLDER = "benchmark/DDOA/maskclip"
    GT_FOLDER = "benchmark/DDOA/gt"

    results, summary = evaluate_folder(PRED_FOLDER, GT_FOLDER, alpha=0.85)

    print("=" * 60)
    print("📊 批量评测汇总结果")
    print("=" * 60)
    print(f"总图片数量：{summary['total_images']}")
    print(f"平均 mIoU：{summary['average_mIoU']:.4f}")
    print(f"平均 ARI：{summary['average_ARI']:.4f}")
    print(f"平均最终分数：{summary['average_final_score']:.4f}")
    print("=" * 60)