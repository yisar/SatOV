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
from kornia.filters import guided_blur

from model import DenseClip


# ======================== LPOSS 核心算法（增强版） ========================
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


def dfs_search(L, Y, tol=1e-6, maxiter=100):
    out = s_linalg.cg(L, Y, rtol=tol, maxiter=maxiter)[0]
    return out


def perform_lp(L, preds):
    preds_np = preds.cpu().numpy()
    N, C = preds_np.shape
    lp_preds = np.zeros((N, C), dtype=np.float32)
    for c in range(C):
        y = preds_np[:, c]
        lp_preds[:, c] = dfs_search(L, y)
    return torch.from_numpy(lp_preds)


def get_pixel_connections(img, neigh=1):
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
    rows_np = rows.cpu().numpy()
    cols_np = cols.cpu().numpy()
    data_np = data.cpu().numpy()
    W = csr_matrix((data_np, (rows_np, cols_np)), shape=(N, N))
    Wn = normalize_connection_graph(W)
    L = eye(N, format="csr") - alpha * Wn
    return L


def lposs_plus(img, preds, tau=0.01, alpha=0.95, r=13):
    preds = preds[0]
    num_classes, h_img, w_img = preds.shape
    preds_flat = preds.permute(1, 2, 0).reshape(h_img * w_img, -1)

    rows, cols, pixel_pixel_data, _ = get_pixel_connections(img, neigh=r // 2)
    pixel_pixel_data = torch.exp(-torch.sqrt(pixel_pixel_data) / tau)

    L = get_laplacian(rows, cols, pixel_pixel_data, preds_flat.shape[0], alpha=alpha)
    lp_preds = perform_lp(L, preds_flat)
    return lp_preds.reshape(h_img, w_img, num_classes).permute(2, 0, 1).unsqueeze(0)


# ======================== 类别保持引导滤波（只平滑不改变类别） ========================
def class_preserving_guided_filter(
    probs, guide_img, radius=2, eps=1e-5, smooth_strength=0.3
):
    """
    引导滤波但保证每个像素的 argmax 类别不变
    probs: (C, H, W) 概率图
    guide_img: (3, H, W) 原始RGB图像 (范围 [0,1])
    radius: 滤波半径
    eps: 正则化参数
    smooth_strength: 平滑强度 (0~1)，0 表示完全保留原始，1 表示完全使用滤波结果（但会强制类别不变）
    返回: (C, H, W) 优化后的概率图，类别不变
    """
    C, H, W = probs.shape
    # 原始类别
    orig_class = probs.argmax(dim=0)  # (H, W)

    # 对概率图进行引导滤波
    probs_tensor = probs.unsqueeze(0)  # (1, C, H, W)
    guide_tensor = guide_img.unsqueeze(0)  # (1, 3, H, W)
    filtered = guided_blur(probs_tensor, guide_tensor, radius, eps)
    filtered = filtered.squeeze(0)  # (C, H, W)

    # 混合原始与滤波结果
    blended = (1 - smooth_strength) * probs + smooth_strength * filtered

    # 强制类别不变：对于每个像素，如果混合后的 argmax 不等于原始类别，则提升原始类别的概率
    new_class = blended.argmax(dim=0)
    mask_changed = new_class != orig_class  # (H, W)
    if mask_changed.any():
        # 对发生变化的像素，将原始类别的概率增加到足够大（使其成为最大值）
        # 方法：将 blended 中原始类别的概率设为该像素所有类别概率的最大值 + 一个小的裕度
        for i in range(C):
            # 只处理那些原始类别为 i 且发生变化的像素
            mask_i = (orig_class == i) & mask_changed
            if not mask_i.any():
                continue
            # 当前 blended 在该像素上的最大值
            max_vals = blended.max(dim=0, keepdim=True)[0]  # (1, H, W)
            # 将原始类别的概率设置为 max_vals + 0.1（确保成为新的最大）
            blended[i, mask_i] = max_vals[0, mask_i] + 0.1
        # 重新归一化
        blended = blended / blended.sum(dim=0, keepdim=True).clamp(min=1e-6)

    return blended


def refine_with_guided_filter(
    full_probs, rgb_tensor, radius=2, eps=1e-5, smooth_strength=0.3
):
    """
    全图引导滤波后处理（类别保持）
    """
    probs_cpu = full_probs.cpu()
    rgb_cpu = rgb_tensor.squeeze(0).cpu()
    refined = class_preserving_guided_filter(
        probs_cpu, rgb_cpu, radius, eps, smooth_strength
    )
    return refined


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
    parser.add_argument("--filename", type=str, default="./asset/img3.jpg")
    parser.add_argument("--window_size", type=int, default=224)
    parser.add_argument("--stride", type=int, default=112)
    parser.add_argument(
        "--refine",
        action="store_true",
        default=True,
        help="Apply class-preserving guided filter after LPOSS",
    )
    return parser.parse_args()


# ======================== 主函数 ========================
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
    n_classes = len(classnames)
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

        full_probs = torch.zeros((n_classes, h, w), device=device)
        weight_sum = torch.zeros((1, h, w), device=device)
        g_mask = get_gaussian_mask(win).to(device)

        y_steps = list(range(0, h - win, stride)) + [h - win]
        x_steps = list(range(0, w - win, stride)) + [w - win]

        print(f">>> 滑动窗口推理 (每窗口内 LPOSS+): {len(y_steps)}x{len(x_steps)}")
        for y in y_steps:
            for x in x_steps:
                crop = raw_image.crop((x, y, x + win, y + win))

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

                img_crop = img_tensor[:, :, y : y + win, x : x + win]
                refined_probs = lposs_plus(img_crop, probs, tau=0.01, alpha=0.95, r=13)
                refined_probs = refined_probs.squeeze(0).to(device)

                full_probs[:, y : y + win, x : x + win] += refined_probs * g_mask
                weight_sum[:, y : y + win, x : x + win] += g_mask

        full_probs /= weight_sum.clamp(min=1e-6)

        # ---------- 类别保持引导滤波后处理 ----------
        if args.refine:
            print(">>> 应用类别保持引导滤波（只平滑边缘，不改变分类）...")
            refined = refine_with_guided_filter(
                full_probs,
                img_tensor,
                radius=2,
                eps=1e-5,
                smooth_strength=0.4,  # 平滑强度，可调节（0~1），0.4表示40%滤波结果+60%原始
            )
            full_probs = refined.to(device)

        # 可视化
        max_idx = full_probs.cpu().argmax(dim=0).numpy()
        masks = torch.stack([torch.from_numpy(max_idx == i) for i in range(n_classes)])
        seg_result = draw_segmentation_masks(
            img_uint8, masks, colors=custom_palette, alpha=1.0
        )

        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        ax[0].imshow(raw_image)
        ax[0].set_title("Original")
        ax[0].axis("off")

        title_suffix = " + ClassPreservingGF" if args.refine else ""
        ax[1].imshow(seg_result.permute(1, 2, 0).numpy())
        ax[1].set_title(f"LPOSS+ (per‑window){title_suffix}")
        ax[1].axis("off")

        patches = [
            mpatches.Patch(color=legend_colors[i], label=classnames[i])
            for i in range(n_classes)
        ]
        fig.legend(handles=patches, loc="center right", title="Classes")
        plt.subplots_adjust(right=0.88)
        plt.show()


if __name__ == "__main__":
    main()
