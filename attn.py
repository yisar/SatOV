import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

from model import CLIPResQQ


def parse_args():
    parser = argparse.ArgumentParser(description="可视化目标类别的归一化概率热力图")
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", type=str, default=default_device)
    parser.add_argument("--filename", type=str, required=True, help="单张图片路径")
    parser.add_argument("--output_dir", type=str, default="./attn_output", help="输出结果文件夹")
    parser.add_argument("--target_class", type=str, default="road",
                        help="要可视化的目标类别名称")
    parser.add_argument("--window_size", type=int, default=224, help="CLIP窗口大小")
    parser.add_argument("--stride", type=int, default=112, help="滑动步长")
    parser.add_argument("--overlay", action="store_true", help="将热力图叠加在原图上")
    parser.add_argument("--alpha", type=float, default=0.5, help="叠加时热力图的透明度")
    parser.add_argument("--show_plot", action="store_true", help="是否弹出可视化窗口")
    return parser.parse_args()


def find_target_index(classnames, target_class):
    """在 classnames 中查找目标类别的索引（支持模糊匹配）"""
    target_lower = target_class.lower().strip()
    for i, name in enumerate(classnames):
        aliases = [a.strip().lower() for a in name.split(",")]
        if target_lower in aliases or target_lower == name.lower():
            return i
    for i, name in enumerate(classnames):
        if target_lower in name.lower():
            return i
    return -1


@torch.no_grad()
def visualize_class_prob(
    img_path: str,
    output_dir: str,
    model,
    device: str,
    window_size: int,
    stride: int,
    classnames: list,
    target_class: str,
    overlay: bool = False,
    alpha: float = 0.5,
    show_plot: bool = False,
):
    win = window_size
    clip_norm = transforms.Normalize((0.4814, 0.4578, 0.4082), (0.2686, 0.2613, 0.2757))

    target_idx = find_target_index(classnames, target_class)
    if target_idx < 0:
        print(f"错误：在类别列表中找不到 '{target_class}'")
        print(f"可用类别：{classnames}")
        return
    print(f">>> 目标类别: '{target_class}' -> 索引 {target_idx} ({classnames[target_idx]})")

    os.makedirs(output_dir, exist_ok=True)

    with Image.open(img_path, "r").convert("RGB") as raw_image:
        w, h = raw_image.size
        img_np = np.array(raw_image)

        all_probs = []
        all_features = []
        all_positions = []

        y_steps = list(range(0, h - win, stride)) + [h - win]
        x_steps = list(range(0, w - win, stride)) + [w - win]
        print(f">>> {os.path.basename(img_path)}: 滑动窗口 {len(y_steps)}x{len(x_steps)} 切片")

        for y in y_steps:
            for x in x_steps:
                crop = raw_image.crop((x, y, x + win, y + win))
                input_clip = transforms.Compose([
                    transforms.Resize((win, win)),
                    transforms.ToTensor(),
                    clip_norm,
                ])(crop).unsqueeze(0).to(device)

                input_guide = transforms.Compose([
                    transforms.Resize((win * 2, win * 2)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])(crop).unsqueeze(0).to(device)

                feat = model.extract_patch_features(input_clip)
                all_features.append(feat.squeeze(0).cpu())

                output = model(input_clip, hr_guide=input_guide)
                output = F.interpolate(output, size=(win, win), mode='bilinear')
                probs = F.softmax(output, dim=1).squeeze(0)
                all_probs.append(probs.cpu())
                all_positions.append((y, x))

        # ================= GLA 注意力计算 =================
        device = torch.device(device)
        all_features = [f.to(device) for f in all_features]
        all_feats = torch.stack(all_features, dim=0)
        N, D, Hf, Wf = all_feats.shape

        global_anchor = all_feats.mean(dim=(0, 2, 3), keepdim=True)
        global_anchor = F.normalize(global_anchor, dim=1)

        keys = all_feats.flatten(start_dim=2)
        keys = F.normalize(keys, dim=1)

        scores = torch.einsum('d, n d l -> n l', global_anchor.squeeze(), keys)
        attn = F.softmax(scores / 0.3, dim=0)

        attn_maps = attn.reshape(N, 1, Hf, Wf)
        attn_maps_win = F.interpolate(attn_maps, size=(win, win), mode='bilinear')

        C = len(classnames)
        acc_probs = torch.zeros((C, h, w), dtype=torch.float32, device=device)
        acc_weights = torch.zeros((h, w), dtype=torch.float32, device=device)

        hann_1d = torch.hann_window(win, device=device)
        hann_2d = torch.outer(hann_1d, hann_1d)

        for idx, (y, x) in enumerate(all_positions):
            prob = all_probs[idx].to(device)
            attn_map = attn_maps_win[idx]
            attn_map = attn_map * hann_2d
            attn_map = attn_map + 1e-4
            acc_probs[:, y:y+win, x:x+win] += prob * attn_map
            acc_weights[y:y+win, x:x+win] += attn_map.squeeze(0)

        acc_weights = acc_weights.clamp(min=1e-6)
        full_probs = acc_probs / acc_weights

        # 背景抑制
        if classnames[0].lower() in ['background', 'bg']:
            full_probs[0] = full_probs[0] * 0.8
            full_probs = full_probs / full_probs.sum(dim=0, keepdim=True).clamp(min=1e-6)

        # 目标类别的概率图，归一化到 [0, 1]
        target_prob = full_probs[target_idx].cpu().numpy()
        pmin, pmax = target_prob.min(), target_prob.max()
        if pmax - pmin > 1e-8:
            target_prob_norm = (target_prob - pmin) / (pmax - pmin)
        else:
            target_prob_norm = np.zeros_like(target_prob)

        base_name = os.path.splitext(os.path.basename(img_path))[0]
        safe_class = target_class.replace(",", "_").replace(" ", "_")

        # ================= 保存归一化概率热力图（红蓝色系） =================
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        if overlay:
            ax.imshow(img_np)
            im = ax.imshow(target_prob_norm, cmap="RdBu_r", alpha=alpha, vmin=0, vmax=1)
        else:
            im = ax.imshow(target_prob_norm, cmap="RdBu_r", vmin=0, vmax=1)
        ax.set_title(f"Normalized Probability: {classnames[target_idx]}")
        ax.axis("off")
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Normalized Probability")
        plt.tight_layout()

        suffix = "_overlay" if overlay else ""
        out_path = os.path.join(output_dir, f"{base_name}_{safe_class}_prob{suffix}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f">>> 保存: {out_path}")

        # ================= 二分类掩码（红色高亮） =================
        target_mask = (target_prob_norm > 0.5).astype(np.uint8) * 255
        mask_path = os.path.join(output_dir, f"{base_name}_{safe_class}_mask.png")
        Image.fromarray(target_mask).save(mask_path)
        print(f">>> 保存二值掩码: {mask_path}")

        overlay_mask = img_np.copy()
        color = np.array([255, 0, 0], dtype=np.uint8)
        mask_bool = target_mask > 0
        overlay_mask[mask_bool] = (overlay_mask[mask_bool] * 0.5 + color * 0.5).astype(np.uint8)
        overlay_path = os.path.join(output_dir, f"{base_name}_{safe_class}_mask_overlay.png")
        Image.fromarray(overlay_mask).save(overlay_path)
        print(f">>> 保存掩码叠加图: {overlay_path}")

        if show_plot:
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            axes[0].imshow(img_np)
            axes[0].set_title("Original")
            axes[0].axis("off")
            im = axes[1].imshow(target_prob_norm, cmap="RdBu_r", vmin=0, vmax=1)
            axes[1].set_title(f"Normalized Probability: {classnames[target_idx]}")
            axes[1].axis("off")
            plt.colorbar(im, ax=axes[1], fraction=0.046)
            plt.tight_layout()
            plt.show()


def main():
    args = parse_args()
    print(f"使用设备: {args.device}")

    classnames = [
        "background",
        "bareland,barren,ground",
        "pavement",
        "road",
        "water,river,pool",
        "tree,forest,vegetation",
        "grass",
        "cropland,field",
        "building,roof,house",
    ]

    if not os.path.exists(args.filename):
        print(f"错误：图片不存在 {args.filename}")
        return

    model = CLIPResQQ("ViT-B-16", classnames, device=args.device, upsampler="aaa")
    model.eval()

    visualize_class_prob(
        img_path=args.filename,
        output_dir=args.output_dir,
        model=model,
        device=args.device,
        window_size=args.window_size,
        stride=args.stride,
        classnames=classnames,
        target_class=args.target_class,
        overlay=args.overlay,
        alpha=args.alpha,
        show_plot=args.show_plot,
    )


if __name__ == "__main__":
    main()