import argparse
from pathlib import Path
import sys
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

root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))
from model import CLIPResQQ


# ======================== LPOSS CORE ========================

def normalize_connection_graph(G):
    W = csr_matrix(G)
    W = W - diags(W.diagonal(), 0)
    S = W.sum(axis=1).A1
    S[S == 0] = 1
    D = 1.0 / np.sqrt(S)
    D[np.isnan(D)] = 0
    D[np.isinf(D)] = 0
    D_mh = diags(D, 0)
    return D_mh @ W @ D_mh


def dfs_search(L, Y, tol=1e-6, maxiter=10):
    return s_linalg.cg(L, Y, rtol=tol, maxiter=maxiter)[0]


# ======================== 🔥 CRF-ENHANCED LPOSS ========================

def perform_lp(L, preds):
    """
    LPOSS + minimal CRF-style correction (NO architecture change)
    """
    preds_np = preds.cpu().numpy()
    N, C = preds_np.shape

    lp_preds = np.zeros((N, C), dtype=np.float32)

    # =========================
    # 1. 原 LPOSS（完全保留）
    # =========================
    for c in range(C):
        lp_preds[:, c] = dfs_search(L, preds_np[:, c])

    # =========================
    # 2. CRF-style unary anchor（防过平滑）
    # =========================
    unary = preds_np
    lp_preds = 0.85 * lp_preds + 0.15 * unary

    # =========================
    # 3. CRF核心：label competition（纠错能力来源）
    # =========================
    lp_preds = np.exp(lp_preds)
    lp_preds = lp_preds / (lp_preds.sum(axis=1, keepdims=True) + 1e-8)

    return torch.from_numpy(lp_preds.astype(np.float32))


# ======================== GRAPH CONSTRUCTION ========================

def get_pixel_connections(img, neigh=1):
    img = img[0]
    img_lab = rgb_to_lab(img).permute(1, 2, 0)
    img_lab = img_lab / torch.tensor([100, 128, 128], device=img.device)

    H, W, _ = img_lab.shape

    coords = torch.stack(torch.meshgrid(
        torch.arange(H, device=img.device),
        torch.arange(W, device=img.device),
        indexing='ij'
    ), -1).reshape(-1, 2)

    rows, cols, vals = [], [], []

    for dy, dx in product(range(-neigh, neigh + 1), range(-neigh, neigh + 1)):
        if dy == 0 and dx == 0:
            continue

        ny = coords[:, 0] + dy
        nx = coords[:, 1] + dx

        mask = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)

        i = coords[mask][:, 0] * W + coords[mask][:, 1]
        j = ny[mask] * W + nx[mask]

        ci = img_lab.reshape(-1, 3)[i]
        cj = img_lab.reshape(-1, 3)[j]

        diff = ((ci - cj) ** 2).sum(-1)

        # =========================
        # edge-aware weakening (minimal CRF flavor)
        # =========================
        w = torch.exp(-torch.sqrt(diff) / 0.01)
        w = w * torch.exp(-diff / 0.02)

        rows.append(i)
        cols.append(j)
        vals.append(w)

    rows = torch.cat(rows).cpu().numpy()
    cols = torch.cat(cols).cpu().numpy()
    vals = torch.cat(vals).cpu().numpy()

    return rows, cols, vals


def get_laplacian(rows, cols, data, N, alpha=0.85):
    W = csr_matrix((data, (rows, cols)), shape=(N, N))
    Wn = normalize_connection_graph(W)
    L = eye(N, format="csr") - alpha * Wn
    return L


# ======================== LPOSS + CRF WRAPPER ========================

def lposs_plus(img, preds, r=13):
    preds = preds[0]
    C, H, W = preds.shape

    preds_flat = preds.permute(1, 2, 0).reshape(H * W, C)

    rows, cols, data = get_pixel_connections(img, neigh=r // 2)
    L = get_laplacian(rows, cols, data, H * W, alpha=0.85)

    refined = perform_lp(L, preds_flat)

    return refined.reshape(H, W, C).permute(2, 0, 1).unsqueeze(0)


# ======================== UTILS ========================

def get_gaussian_mask(size, sigma=0.4):
    coords = torch.arange(size).float() - (size - 1) / 2
    g = torch.exp(-(coords**2) / (2 * (sigma * size) ** 2))
    mask = g.view(-1, 1) @ g.view(1, -1)
    return mask / mask.max()


def parse_args():
    parser = argparse.ArgumentParser()
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", type=str, default=default_device)
    parser.add_argument("--filename", type=str, default="data/DDOA/origin/_P1886.png")
    parser.add_argument("--window_size", type=int, default=224)
    parser.add_argument("--stride", type=int, default=112)
    return parser.parse_args()


# ======================== MAIN ========================

@torch.no_grad()
def main():
    args = parse_args()
    win, stride = args.window_size, args.stride
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

    model = CLIPResQQ("ViT-B-16", classnames, device=device, only_clear=True)
    model.eval()

    clip_norm = transforms.Normalize((0.4814, 0.4578, 0.4082),
                                     (0.2686, 0.2613, 0.2757))

    with Image.open(args.filename).convert("RGB") as raw_image:
        w, h = raw_image.size
        img_tensor = TF.to_tensor(raw_image).unsqueeze(0).to(device)
        img_uint8 = TF.to_tensor(raw_image).to(torch.uint8)

        full_probs = torch.zeros((len(classnames), h, w), device=device)
        weight_sum = torch.zeros((1, h, w), device=device)
        g_mask = get_gaussian_mask(win).to(device)

        y_steps = list(range(0, h - win + 1, stride)) + [h - win] if h > win else [0]
        x_steps = list(range(0, w - win + 1, stride)) + [w - win] if w > win else [0]

        print(f">>> sliding window: {len(y_steps)} x {len(x_steps)}")

        for y in y_steps:
            for x in x_steps:

                y_end = min(y + win, h)
                x_end = min(x + win, w)

                crop_h = y_end - y
                crop_w = x_end - x

                crop = raw_image.crop((x, y, x_end, y_end))

                input_clip = transforms.Compose([
                    transforms.Resize((win, win)),
                    transforms.ToTensor(),
                    clip_norm,
                ])(crop).unsqueeze(0).to(device)

                input_guide = transforms.Compose([
                    transforms.Resize((win * 2, win * 2)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225]),
                ])(crop).unsqueeze(0).to(device)

                output = model(input_clip, hr_guide=input_guide)
                output = F.interpolate(output, size=(win, win), mode="bilinear")
                probs = F.softmax(output, dim=1)

                probs_crop = F.interpolate(probs, size=(crop_h, crop_w), mode="bilinear")
                img_crop = img_tensor[:, :, y:y_end, x:x_end]

                refined = lposs_plus(img_crop, probs_crop).squeeze(0).to(device)

                g_crop = g_mask[:crop_h, :crop_w]

                full_probs[:, y:y_end, x:x_end] += refined * g_crop
                weight_sum[:, y:y_end, x:x_end] += g_crop

        full_probs /= weight_sum.clamp(min=1e-6)

        mask = full_probs.cpu().argmax(dim=0)

        masks = torch.stack([mask == i for i in range(len(classnames))])

        seg = draw_segmentation_masks(
            img_uint8,
            masks,
            colors=custom_palette,
            alpha=1.0
        )

        Image.fromarray(seg.permute(1, 2, 0).numpy()).save(
            "data/DDOA/origin/_P1886_lposs.png"
        )

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(raw_image)
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(seg.permute(1, 2, 0).numpy())
        plt.title("LPOSS + CRF correction (minimal change)")
        plt.axis("off")

        plt.show()


if __name__ == "__main__":
    main()