import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import adjusted_rand_score


# ===============================
# 1️⃣ 自动读取 mask（支持灰度 / RGB）
# ===============================
def load_mask(path):
    img = Image.open(path)

    # RGB 彩色分割图
    if img.mode == "RGB":
        return rgb_to_label(img)
    else:
        # 灰度图直接当 label
        return np.array(img)


# ===============================
# 2️⃣ RGB → label（关键）
# ===============================
def rgb_to_label(img):
    img = np.array(img)

    h, w, _ = img.shape
    label = np.zeros((h, w), dtype=np.int32)

    # 找所有颜色
    colors = np.unique(img.reshape(-1, 3), axis=0)

    # 颜色 → 类别ID
    for idx, color in enumerate(colors):
        mask = np.all(img == color, axis=-1)
        label[mask] = idx

    return label


# ===============================
# 3️⃣ IoU matrix
# ===============================
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


# ===============================
# 4️⃣ Hungarian mIoU
# ===============================
def hungarian_miou(pred, gt):
    iou_matrix = compute_iou_matrix(pred, gt)

    if iou_matrix.size == 0:
        return 0.0

    cost = 1 - iou_matrix
    row_ind, col_ind = linear_sum_assignment(cost)

    return iou_matrix[row_ind, col_ind].mean()


# ===============================
# 5️⃣ ARI
# ===============================
def compute_ari(pred, gt):
    return adjusted_rand_score(gt.flatten(), pred.flatten())


# ===============================
# 6️⃣ 总评测
# ===============================
def evaluate_segmentation(pred, gt, alpha=0.5):
    miou = hungarian_miou(pred, gt)
    ari = compute_ari(pred, gt)

    return {
        "mIoU_hungarian": miou,
        "ARI": ari,
        "final_score": alpha * miou + (1 - alpha) * ari
    }


# ===============================
# 7️⃣ 主程序（输入图片路径）
# ===============================
pred_path = "pred.png"
gt_path = "gt.png"

pred = load_mask(pred_path)
gt = load_mask(gt_path)

# 🔥 你要加的 debug（已加入）
print("pred unique:", np.unique(pred))
print("gt unique:", np.unique(gt))
print("颜色数量（pred）:", len(np.unique(pred)))
print("颜色数量（gt）:", len(np.unique(gt)))

result = evaluate_segmentation(pred, gt)

print(result)