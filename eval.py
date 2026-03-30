import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import adjusted_rand_score


def load_mask(path):
    """
    读取分割mask图片 → numpy数组
    """
    img = Image.open(path)

    # 转成单通道（非常关键！）
    img = img.convert("L")

    return np.array(img)


def compute_iou_matrix(pred, gt):
    pred = pred.flatten()
    gt = gt.flatten()

    pred_ids = np.unique(pred)
    gt_ids = np.unique(gt)

    iou_matrix = np.zeros((len(gt_ids), len(pred_ids)))

    for i, g in enumerate(gt_ids):
        gt_mask = (gt == g)
        for j, p in enumerate(pred_ids):
            pred_mask = (pred == p)

            inter = np.sum(gt_mask & pred_mask)
            union = np.sum(gt_mask | pred_mask)

            if union > 0:
                iou_matrix[i, j] = inter / union

    return iou_matrix


def hungarian_miou(pred, gt):
    iou_matrix = compute_iou_matrix(pred, gt)

    if iou_matrix.size == 0:
        return 0.0

    cost = 1 - iou_matrix
    row_ind, col_ind = linear_sum_assignment(cost)

    return iou_matrix[row_ind, col_ind].mean()


def compute_ari(pred, gt):
    return adjusted_rand_score(gt.flatten(), pred.flatten())


def evaluate_segmentation(pred, gt, alpha=0.5):
    miou = hungarian_miou(pred, gt)
    ari = compute_ari(pred, gt)

    return {
        "mIoU_hungarian": miou,
        "ARI": ari,
        "final_score": alpha * miou + (1 - alpha) * ari
    }


# ====== 这里是关键：输入图片路径 ======

pred_path = "pred.png"
gt_path = "gt.png"

pred = load_mask(pred_path)
gt = load_mask(gt_path)

result = evaluate_segmentation(pred, gt)

print(result)