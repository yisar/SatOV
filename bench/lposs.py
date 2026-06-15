import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torchvision import transforms
from torchvision.utils import draw_segmentation_masks

from itertools import product
from scipy.sparse import csr_matrix, eye, diags
from scipy.sparse import linalg as s_linalg
from kornia.color import rgb_to_lab

from model import DenseClip


# ======================== LPOSS 核心算法（纯 CPU 版本） ========================
def make_input_divisible(x: torch.Tensor, patch_size=16) -> torch.Tensor:
    B, _, H_0, W_0 = x.shape
    pad_w = (patch_size - W_0 % patch_size) % patch_size
    pad_h = (patch_size - H_0 % patch_size) % patch_size
    x = F.pad(x, (0, pad_w, 0, pad_h), value=0)
    return x

def reshape_windows(x):
    height_width = [(y.shape[0], y.shape[1]) for y in x]
    dim = x[0].shape[-1]
    x = [torch.reshape(y, (-1, dim)) for y in x]
    return torch.cat(x, dim=0), height_width


def normalize_connection_graph(G):
    """归一化亲和矩阵 (对称归一化)"""
    W = csr_matrix(G)
    W = W - diags(W.diagonal(), 0)
    S = W.sum(axis=1).A1
    S[S == 0] = 1
    D = 1.0 / np.sqrt(S)
    D[np.isnan(D)] = 0
    D[np.isinf(D)] = 0
    D_mh = diags(D, 0)
    Wn = D_mh @ W @ D_mh
    return Wn


def dfs_search(L, Y, tol=1e-6, maxiter=10):
    """共轭梯度法求解 L·x = Y"""
    out = s_linalg.cg(L, Y, rtol=tol, maxiter=maxiter)[0]
    return out


def perform_lp(L, preds):
    """
    对每个类别执行标签传播
    L: scipy.sparse 拉普拉斯矩阵
    preds: torch.Tensor (N_pixels, C)，位于 CPU
    返回: torch.Tensor (N_pixels, C)
    """
    preds_np = preds.cpu().numpy()
    N, C = preds_np.shape
    lp_preds = np.zeros((N, C), dtype=np.float32)
    for c in range(C):
        y = preds_np[:, c]
        lp_preds[:, c] = dfs_search(L, y)
    return torch.from_numpy(lp_preds)


def get_pixel_connections(img, neigh=1):
    """
    构建像素间连接（基于 LAB 颜色空间）
    img: (1,3,H,W) 归一化 RGB 图像
    返回: rows, cols (连接索引), pixel_pixel_data (相似度权重), locs (未使用)
    """
    img = img[0, ...]
    img_lab = rgb_to_lab(img)
    img_lab = img_lab.permute((1, 2, 0))
    img_lab /= torch.tensor([100, 128, 128], device=img.device)
    img_h, img_w, _ = img_lab.shape
    img_lab = img_lab.reshape((img_h * img_w, -1))

    idx = torch.arange(img_h * img_w).to(img.device)
    loc_h = idx // img_w
    loc_w = idx % img_w
    locs = torch.stack((loc_h, loc_w), 1)

    rows, cols = [], []
    for mov in product(range(-neigh, neigh + 1), range(-neigh, neigh + 1)):
        if mov == (0, 0):
            continue
        new_locs = locs + torch.tensor(mov).to(img.device)
        mask = (
            (new_locs[:, 0] >= 0)
            & (new_locs[:, 1] >= 0)
            & (new_locs[:, 0] < img_h)
            & (new_locs[:, 1] < img_w)
        )
        rows.append(torch.where(mask)[0])
        col = new_locs[mask]
        col = col[:, 0] * img_w + col[:, 1]
        cols.append(col)

    rows = torch.cat(rows)
    cols = torch.cat(cols)
    pixel_pixel_data = ((img_lab[rows] - img_lab[cols]) ** 2).sum(dim=-1)
    return rows, cols, pixel_pixel_data, locs


def get_laplacian(rows, cols, data, N, alpha=0.95):
    """构建拉普拉斯矩阵 L = I - alpha * Wn"""
    rows_np = rows.cpu().numpy()
    cols_np = cols.cpu().numpy()
    data_np = data.cpu().numpy()
    W = csr_matrix((data_np, (rows_np, cols_np)), shape=(N, N))
    Wn = normalize_connection_graph(W)
    L = eye(N, format="csr") - alpha * Wn
    return L


def lposs_plus(img, preds, tau=0.01, alpha=0.95, r=13):
    """
    对单个图像块执行 LPOSS+
    img: (1,3,H,W) 归一化 RGB
    preds: (1,C,H,W) 原始概率图 (softmax 后)
    返回: (1,C,H,W) 优化后的概率图
    """
    preds = preds[0]
    num_classes, h_img, w_img = preds.shape
    preds_flat = preds.permute(1, 2, 0).reshape(h_img * w_img, -1)

    rows, cols, pixel_pixel_data, _ = get_pixel_connections(img, neigh=r // 2)
    pixel_pixel_data = torch.exp(-torch.sqrt(pixel_pixel_data) / tau)

    L = get_laplacian(rows, cols, pixel_pixel_data, preds_flat.shape[0], alpha=alpha)
    lp_preds = perform_lp(L, preds_flat)
    return lp_preds.reshape(h_img, w_img, num_classes).permute(2, 0, 1).unsqueeze(0)


# ======================== 工具函数 ========================
def get_gaussian_mask(size, sigma=0.4):
    coords = torch.arange(size).float() - (size - 1) / 2
    g = torch.exp(-(coords**2) / (2 * (sigma * size) ** 2))
    mask = g.view(-1, 1) @ g.view(1, -1)
    return mask / mask.max()


def parse_args():
    parser = argparse.ArgumentParser()
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", type=str, default=default_device)
    parser.add_argument("--filename", type=str, default="./asset/img2.jpg")
    parser.add_argument("--window_size", type=int, default=224)
    parser.add_argument("--stride", type=int, default=112)
    return parser.parse_args()


# ======================== 【已修复】主函数：保持原图尺寸 ========================
@torch.no_grad()
def main():
    args = parse_args()
    win = args.window_size
    stride = args.stride
    device = args.device

    classnames = [
        "background",
        "bareland,barren",
        "pavement",
        "road",
        "water,river",
        "tree,forest",
        "grass",
        "cropland,field",
        "building,roof,house",
    ]
    custom_palette = [
        (68, 1, 84),
        (72, 40, 120),
        (62, 74, 137),
        (49, 104, 142),
        (38, 130, 142),
        (31, 158, 137),
        (73, 193, 110),
        (160, 218, 57),
        (253, 231, 37),
    ]
    legend_colors = [tuple(c / 255 for c in color) for color in custom_palette]

    model = DenseClip("ViT-B-16", classnames, device=device)
    model.eval()

    clip_norm = transforms.Normalize((0.4814, 0.4578, 0.4082), (0.2686, 0.2613, 0.2757))

    with Image.open(args.filename).convert("RGB") as raw_image:
        w, h = raw_image.size
        img_tensor = TF.to_tensor(raw_image).unsqueeze(0).to(device)
        img_uint8 = TF.to_tensor(raw_image).to(torch.uint8)

        full_probs = torch.zeros((len(classnames), h, w), device=device)
        weight_sum = torch.zeros((1, h, w), device=device)
        g_mask = get_gaussian_mask(win).to(device)

        y_steps = list(range(0, h - win + 1, stride)) + [h - win] if h > win else [0]
        x_steps = list(range(0, w - win + 1, stride)) + [w - win] if w > win else [0]

        print(f">>> 滑动窗口推理 (每窗口内 LPOSS): {len(y_steps)}x{len(x_steps)}")
        for y in y_steps:
            for x in x_steps:
                print(f"{x},{y}")
                y_end = min(y + win, h)
                x_end = min(x + win, w)
                crop_h = y_end - y
                crop_w = x_end - x

                crop = raw_image.crop((x, y, x_end, y_end))

                input_clip = (
                    transforms.Compose(
                        [
                            transforms.Resize((win, win)),
                            transforms.ToTensor(),
                            clip_norm,
                        ]
                    )(crop)
                    .unsqueeze(0)
                    .to(device)
                )

                input_guide = (
                    transforms.Compose(
                        [
                            transforms.Resize((win * 2, win * 2)),
                            transforms.ToTensor(),
                            transforms.Normalize(
                                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                            ),
                        ]
                    )(crop)
                    .unsqueeze(0)
                    .to(device)
                )

                output = model(input_clip, hr_guide=input_guide)
                output = F.interpolate(output, size=(win, win), mode="bilinear")
                probs = F.softmax(output, dim=1)

                probs_crop = F.interpolate(
                    probs, size=(crop_h, crop_w), mode="bilinear"
                )
                img_crop = img_tensor[:, :, y:y_end, x:x_end]

                refined_probs = lposs_plus(
                    img_crop, probs_crop, tau=0.01, alpha=0.95, r=13
                )
                refined_probs = refined_probs.squeeze(0).to(device)

                g_crop = g_mask[:crop_h, :crop_w]
                full_probs[:, y:y_end, x:x_end] += refined_probs * g_crop
                weight_sum[:, y:y_end, x:x_end] += g_crop

        full_probs /= weight_sum.clamp(min=1e-6)

        max_idx = full_probs.cpu().argmax(dim=0).numpy()
        masks = torch.stack(
            [torch.from_numpy(max_idx == i) for i in range(len(classnames))]
        )
        seg_result = draw_segmentation_masks(
            img_uint8, masks, colors=custom_palette, alpha=1.0
        )
        seg_result_pil = TF.to_pil_image(seg_result)
        seg_result_pil.save(args.filename.replace('origin', 'lposs'))

        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        ax[0].imshow(raw_image)
        ax[0].set_title("Original")
        ax[0].axis("off")

        ax[1].imshow(seg_result.permute(1, 2, 0).numpy())
        ax[1].set_title("LPOSS+ (per‑window refinement, CPU)")
        ax[1].axis("off")

        patches = [
            mpatches.Patch(color=legend_colors[i], label=classnames[i])
            for i in range(len(classnames))
        ]
        fig.legend(handles=patches, loc="center right", title="Classes")
        plt.subplots_adjust(right=0.88)
        plt.show()


if __name__ == "__main__":
    main()
