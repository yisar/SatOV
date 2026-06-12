import os
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment


# =====================================================
# 固定调色板
# =====================================================
CUSTOM_PALETTE = np.array(
    [
        (68, 1, 84),
        (72, 40, 120),
        (62, 74, 137),
        (49, 104, 142),
        (38, 130, 142),
        (31, 158, 137),
        (73, 193, 110),
        (160, 218, 57),
        (253, 231, 37),
    ],
    dtype=np.int16,
)

NUM_CLASSES = len(CUSTOM_PALETTE)


# =====================================================
# RGB -> 最近Palette类别
# =====================================================
def rgb_to_label(img):

    img = np.asarray(img, dtype=np.int16)

    h, w, _ = img.shape

    pixels = img.reshape(-1, 3)

    # (N,9,3)
    diff = pixels[:, None, :] - CUSTOM_PALETTE[None, :, :]

    dist = np.sum(diff * diff, axis=2)

    labels = np.argmin(dist, axis=1)

    return labels.reshape(h, w).astype(np.int32)


# =====================================================
# Load Mask
# =====================================================
def load_mask(path):

    img = Image.open(path).convert("RGB")

    return rgb_to_label(img)


# =====================================================
# Resize
# =====================================================
def resize_pred_to_gt(pred, gt):

    if pred.shape == gt.shape:
        return pred

    pred_img = Image.fromarray(pred.astype(np.uint8))

    pred_img = pred_img.resize(
        (gt.shape[1], gt.shape[0]),
        Image.NEAREST,
    )

    return np.array(pred_img)


# =====================================================
# Confusion Matrix
# =====================================================
def compute_confusion_matrix(pred, gt):

    valid = (pred >= 0) & (gt >= 0)

    pred = pred[valid]
    gt = gt[valid]

    confusion = np.bincount(
        gt * NUM_CLASSES + pred,
        minlength=NUM_CLASSES * NUM_CLASSES,
    ).reshape(NUM_CLASSES, NUM_CLASSES)

    return confusion


# =====================================================
# Hungarian Matching
# =====================================================
def hungarian_match(confusion):

    gt_sum = confusion.sum(axis=1, keepdims=True)

    pred_sum = confusion.sum(axis=0, keepdims=True)

    intersection = confusion

    union = gt_sum + pred_sum - intersection

    iou_matrix = intersection / np.maximum(union, 1)

    row_ind, col_ind = linear_sum_assignment(1.0 - iou_matrix)

    return row_ind, col_ind, iou_matrix


# =====================================================
# Hungarian mIoU
# =====================================================
def compute_miou(pred, gt):

    pred = resize_pred_to_gt(pred, gt)

    confusion = compute_confusion_matrix(pred, gt)

    row_ind, col_ind, iou_matrix = hungarian_match(confusion)

    matched_iou = iou_matrix[row_ind, col_ind]

    valid = matched_iou > 0

    if valid.sum() == 0:
        return 0.0

    return float(matched_iou[valid].mean())


# =====================================================
# Hungarian Pixel Accuracy
# =====================================================
def compute_pixel_acc(pred, gt):

    pred = resize_pred_to_gt(pred, gt)

    confusion = compute_confusion_matrix(pred, gt)

    row_ind, col_ind, _ = hungarian_match(confusion)

    matched_confusion = confusion[:, col_ind]

    correct = np.trace(matched_confusion)

    total = confusion.sum()

    return float(correct / max(total, 1))


# =====================================================
# Evaluate One
# =====================================================
def evaluate_image(pred, gt):

    return {
        "mIoU": compute_miou(pred, gt),
        "PixelAcc": compute_pixel_acc(pred, gt),
    }


# =====================================================
# Evaluate Folder
# =====================================================
def evaluate_folder(pred_folder, gt_folder):

    pred_files = sorted(
        [
            f
            for f in os.listdir(pred_folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    )

    gt_files = sorted(
        [
            f
            for f in os.listdir(gt_folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    )

    assert len(pred_files) == len(gt_files)

    print(f"\nFound {len(pred_files)} images\n")

    all_miou = []
    all_acc = []

    for idx, (pred_name, gt_name) in enumerate(
        zip(pred_files, gt_files),
        1,
    ):
        pred = load_mask(os.path.join(pred_folder, pred_name))

        gt = load_mask(os.path.join(gt_folder, gt_name))

        result = evaluate_image(pred, gt)

        all_miou.append(result["mIoU"])
        all_acc.append(result["PixelAcc"])

        print(f"[{idx}/{len(pred_files)}] {pred_name}")

        print(f"mIoU={result['mIoU']:.4f} | PixelAcc={result['PixelAcc']:.4f}")

    avg_miou = float(np.mean(all_miou))
    avg_acc = float(np.mean(all_acc))

    composite = np.sqrt(avg_miou * avg_acc)

    print("\n" + "=" * 60)

    print(f"Average mIoU      : {avg_miou:.4f}")
    print(f"Average PixelAcc  : {avg_acc:.4f}")
    print(f"Composite Score   : {composite:.4f}")

    print("=" * 60)

    return {
        "mIoU": avg_miou,
        "PixelAcc": avg_acc,
        "CompositeScore": composite,
    }


# =====================================================
# Main
# =====================================================
if __name__ == "__main__":
    PRED_FOLDER = "bench/SSSI/maskclip"
    GT_FOLDER = "bench/SSSI/gt"

    evaluate_folder(
        PRED_FOLDER,
        GT_FOLDER,
    )
