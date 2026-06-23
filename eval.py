import os
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

# =====================================================
# 配置区（可按需修改）
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
        (250, 200, 20),
    ],
    dtype=np.float32,
)
NUM_CLASSES = len(CUSTOM_PALETTE)
IGNORE_LABEL = 255  # 无效忽略像素，无忽略则设为-1
# 当前为模式B：无监督/开放集分割，启用匈牙利类别匹配
USE_HUNGARIAN_MATCH = True

# =====================================================
# RGB图像转类别标签（优化数值精度，防止距离溢出）
# =====================================================
def rgb_to_label(img):
    img = np.asarray(img, dtype=np.float32)
    h, w, _ = img.shape
    pixels = img.reshape(-1, 3)
    # 欧式距离平方
    diff = pixels[:, None, :] - CUSTOM_PALETTE[None, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    labels = np.argmin(dist_sq, axis=1)
    return labels.reshape(h, w).astype(np.int32)

# =====================================================
# 读取掩码RGB图，转为类别索引图
# =====================================================
def load_mask(path):
    img = Image.open(path).convert("RGB")
    return rgb_to_label(img)

# =====================================================
# 将预测掩码缩放至GT分辨率，近邻插值
# =====================================================
def resize_pred_to_gt(pred, gt):
    if pred.shape == gt.shape:
        return pred
    h_gt, w_gt = gt.shape
    pred_img = Image.fromarray(pred.astype(np.uint16))
    pred_img = pred_img.resize((w_gt, h_gt), Image.NEAREST)
    return np.array(pred_img, dtype=np.int32)

# =====================================================
# 计算混淆矩阵，过滤ignore label无效像素
# =====================================================
def compute_confusion_matrix(pred, gt):
    # 筛选有效像素：非忽略标签、类别合法
    valid_mask = (gt != IGNORE_LABEL) & (pred >= 0) & (pred < NUM_CLASSES)
    pred_valid = pred[valid_mask]
    gt_valid = gt[valid_mask]

    if len(pred_valid) == 0:
        return np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    confusion = np.bincount(
        gt_valid * NUM_CLASSES + pred_valid,
        minlength=NUM_CLASSES * NUM_CLASSES,
    ).reshape(NUM_CLASSES, NUM_CLASSES)
    return confusion

# =====================================================
# 匈牙利最优类别匹配（模式B核心逻辑保留）
# 输入混淆矩阵(gt行, pred列)，输出最优映射
# =====================================================
def hungarian_match(confusion):
    gt_sum = confusion.sum(axis=1, keepdims=True)  # 每行GT各类总像素
    pred_sum = confusion.sum(axis=0, keepdims=True)# 每列Pred各类总像素
    intersection = confusion
    union = gt_sum + pred_sum - intersection

    # IoU矩阵，空类union置无穷大，避免除0
    union_safe = np.where(union > 0, union, np.inf)
    iou_matrix = intersection / union_safe

    # 代价矩阵 = 1 - IoU，匈牙利求最小代价匹配
    cost_mat = 1.0 - iou_matrix
    gt_matched_idx, pred_matched_idx = linear_sum_assignment(cost_mat)
    return gt_matched_idx, pred_matched_idx, iou_matrix

# =====================================================
# 模式B：匹配后计算 mIoU（平均IoU）
# =====================================================
def compute_hungarian_miou(pred, gt):
    pred = resize_pred_to_gt(pred, gt)
    conf = compute_confusion_matrix(pred, gt)
    gt_ids, pred_ids, iou_mat = hungarian_match(conf)

    matched_ious = iou_mat[gt_ids, pred_ids]
    # 过滤IoU=0无重叠的无效匹配类别
    valid_ious = matched_ious[matched_ious > 1e-6]
    if len(valid_ious) == 0:
        return 0.0
    return float(np.mean(valid_ious))

# =====================================================
# 模式B：匹配后 mAcc（平均类别召回率，行业标准）
# mAcc = 每类召回率(TP/GT总像素)求平均
# =====================================================
def compute_hungarian_macc(pred, gt):
    pred = resize_pred_to_gt(pred, gt)
    conf = compute_confusion_matrix(pred, gt)
    gt_ids, pred_ids, _ = hungarian_match(conf)

    # 构建匹配后的混淆矩阵：行=GT，列=匹配后的Pred
    matched_conf = conf[:, pred_ids]
    gt_total = matched_conf.sum(axis=1)  # 每一类GT总像素
    tp = matched_conf.diagonal()         # 每类正确匹配像素

    recall = np.zeros(NUM_CLASSES)
    mask = gt_total > 0
    recall[mask] = tp[mask] / gt_total[mask]

    valid_recall = recall[recall > 1e-6]
    if len(valid_recall) == 0:
        return 0.0
    return float(np.mean(valid_recall))

# =====================================================
# 全局像素准确率 PixelAcc（所有像素正确占比）
# =====================================================
def compute_pixel_acc(pred, gt):
    pred = resize_pred_to_gt(pred, gt)
    conf = compute_confusion_matrix(pred, gt)
    gt_ids, pred_ids, _ = hungarian_match(conf)
    matched_conf = conf[:, pred_ids]

    total_pixels = conf.sum()
    correct_pixels = matched_conf.diagonal().sum()
    if total_pixels == 0:
        return 0.0
    return float(correct_pixels / total_pixels)

# =====================================================
# 单张图像全套指标评估
# =====================================================
def evaluate_single_image(pred_mask, gt_mask):
    if USE_HUNGARIAN_MATCH:
        miou = compute_hungarian_miou(pred_mask, gt_mask)
        macc = compute_hungarian_macc(pred_mask, gt_mask)
        pixel_acc = compute_pixel_acc(pred_mask, gt_mask)
    else:
        # 模式A：监督分割一一对应（本次不用）
        raise NotImplementedError("当前仅启用模式B匈牙利匹配")
    return {
        "mIoU": miou,
        "mAcc": macc,
        "PixelAcc": pixel_acc
    }

# =====================================================
# 批量文件夹评估（修复文件匹配bug）
# =====================================================
def evaluate_folder(pred_folder, gt_folder):
    # 读取所有图片并构建文件名映射
    exts = (".png", ".jpg", ".jpeg")
    pred_files = {f.lower(): f for f in os.listdir(pred_folder) if f.lower().endswith(exts)}
    gt_files = {f.lower(): f for f in os.listdir(gt_folder) if f.lower().endswith(exts)}

    # 取交集同名文件，严格匹配
    common_names = sorted(set(pred_files.keys()) & set(gt_files.keys()))
    if len(common_names) == 0:
        raise FileNotFoundError("预测集与真值集无同名图片，请检查文件夹！")

    print(f"\n共匹配到 {len(common_names)} 张评估图像\n")
    all_miou, all_macc, all_pixelacc = [], [], []

    for idx, name in enumerate(common_names, start=1):
        pred_path = os.path.join(pred_folder, pred_files[name])
        gt_path = os.path.join(gt_folder, gt_files[name])

        pred = load_mask(pred_path)
        gt = load_mask(gt_path)
        res = evaluate_single_image(pred, gt)

        all_miou.append(res["mIoU"])
        all_macc.append(res["mAcc"])
        all_pixelacc.append(res["PixelAcc"])

        print(f"[{idx}/{len(common_names)}] {pred_files[name]}")
        print(f"mIoU={res['mIoU']:.4f} | mAcc={res['mAcc']:.4f} | PixelAcc={res['PixelAcc']:.4f}\n")

    # 全局平均指标
    avg_miou = float(np.mean(all_miou))
    avg_macc = float(np.mean(all_macc))
    avg_pixelacc = float(np.mean(all_pixelacc))
    composite = np.sqrt(avg_miou * avg_macc)

    print("=" * 70)
    print(f"全局平均 mIoU        : {avg_miou:.4f}")
    print(f"全局平均 mAcc        : {avg_macc:.4f}")
    print(f"全局像素准确率 PixelAcc : {avg_pixelacc:.4f}")
    print(f"综合得分 sqrt(mIoU*mAcc) : {composite:.4f}")
    print("=" * 70)

    return {
        "avg_mIoU": avg_miou,
        "avg_mAcc": avg_macc,
        "avg_PixelAcc": avg_pixelacc,
        "CompositeScore": composite
    }

# =====================================================
# 入口
# =====================================================
if __name__ == "__main__":
    PRED_FOLDER = "out/UDD6_sat"
    GT_FOLDER = "out/UDD6_gt"
    result = evaluate_folder(PRED_FOLDER, GT_FOLDER)