import argparse
import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torchvision import transforms
from torchvision.utils import draw_segmentation_masks

# 核心后处理包
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import (
    unary_from_softmax,
)

from model import CLIPResQQ


def apply_dense_crf(img_np, probs_np):
    """
    使用 Dense CRF 优化分割结果
    img_np: 原始 RGB 图像 (H, W, 3), np.uint8
    probs_np: 模型输出的概率分布 (C, H, W), np.float32
    """
    C, H, W = probs_np.shape
    d = dcrf.DenseCRF2D(W, H, C)  # 注意这里是 W, H

    # 1. 设置一元势能 (模型给出的分类概率)
    # unary_from_softmax 需要的输入是 (C, H*W)
    unary = unary_from_softmax(probs_np)
    d.setUnaryEnergy(unary)

    # 2. 添加二元势能 - 高斯项 (只考虑位置，去除孤立的小噪点)
    # sxy 控制平滑的强弱，数值越大越平滑
    d.addPairwiseGaussian(
        sxy=(3, 3),
        compat=3,
        kernel=dcrf.DIAG_KERNEL,
        normalization=dcrf.NORMALIZE_SYMMETRIC,
    )

    # 3. 添加二元势能 - 双边项 (考虑位置+颜色，核心：让边缘对齐纹理)
    # sxy: 空间位置标准差; srgb: 颜色标准差 (数值越小，对颜色差异越敏感)
    d.addPairwiseBilateral(
        sxy=(40, 40),
        srgb=(13, 13, 13),
        rgbim=img_np,
        compat=10,
        kernel=dcrf.DIAG_KERNEL,
        normalization=dcrf.NORMALIZE_SYMMETRIC,
    )

    # 执行推理 (迭代 5-10 次即可)
    Q = d.inference(10)

    # 将结果转回 (C, H, W)
    return np.array(Q).reshape((C, H, W))
# uv run infer.py --input_dir ./out/UDD6 --output_dir ./out/UDD6_out
# python infer.py --filename out/UDD6/DJI_0421.jpg --show_plot

@torch.no_grad()
def infer_single_image(
    img_path: str,
    out_save_path: str,
    model,
    device: str,
    window_size: int,
    stride: int,
    classnames: list,
    custom_palette: list,
    use_crf: bool = True,
    show_plot: bool = False
):
    win = window_size
    clip_norm = transforms.Normalize((0.4814, 0.4578, 0.4082), (0.2686, 0.2613, 0.2757))
    legend_colors = [tuple(ch / 255 for ch in rgb) for rgb in custom_palette]

    with Image.open(img_path, "r").convert("RGB") as raw_image:
        w, h = raw_image.size
        img_np = np.array(raw_image)
        img_tensor = TF.to_tensor(raw_image).multiply(255).to(torch.uint8)

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

                # 低分辨率特征 [1, D, Hf, Wf]
                feat = model.extract_patch_features(input_clip)
                all_features.append(feat.squeeze(0).cpu())

                # 高分辨率概率图 [C, win, win]
                output = model(input_clip, hr_guide=input_guide)
                output = F.interpolate(output, size=(win, win), mode='bilinear')
                probs = F.softmax(output, dim=1).squeeze(0)
                all_probs.append(probs.cpu())
                all_positions.append((y, x))

        # =========================================================
        #  GLA 计算（改进版）
        # =========================================================
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        all_features = [f.to(device) for f in all_features]
        all_feats = torch.stack(all_features, dim=0)   # [N, D, Hf, Wf]
        N, D, Hf, Wf = all_feats.shape

        # 1. 全局锚点（归一化）
        global_anchor = all_feats.mean(dim=(0, 2, 3), keepdim=True)  # [1,D,1,1]
        global_anchor = F.normalize(global_anchor, dim=1)            # L2 归一化

        # 2. 展平并归一化 keys
        keys = all_feats.flatten(start_dim=2)  # [N, D, Hf*Wf]
        keys = F.normalize(keys, dim=1)        # 沿 D 维归一化

        # 3. 相似度（点积）
        scores = torch.einsum('d, n d l -> n l', global_anchor.squeeze(), keys)  # [N, Hf*Wf]

        # 4. 温度调整（0.3 使权重更平滑）
        attn = F.softmax(scores / 0.3, dim=0)  # [N, Hf*Wf]

        # 5. 重塑 + 上采样
        attn_maps = attn.reshape(N, 1, Hf, Wf)
        attn_maps_win = F.interpolate(attn_maps, size=(win, win), mode='bilinear')

        C = len(classnames)
        acc_probs = torch.zeros((C, h, w), dtype=torch.float32, device=device)
        acc_weights = torch.zeros((h, w), dtype=torch.float32, device=device)

        # ---- 新增：生成二维汉宁窗 ----
        hann_1d = torch.hann_window(win, device=device)        # [win]
        hann_2d = torch.outer(hann_1d, hann_1d)                # [win, win]
        # ------------------------------

        for idx, (y, x) in enumerate(all_positions):
            prob = all_probs[idx].to(device)          # [C, win, win]
            attn_map = attn_maps_win[idx]             # [1, win, win]
            # ---- 新增：乘以汉宁窗（空间平滑） ----
            attn_map = attn_map * hann_2d
            # --------------------------------------
            attn_map = attn_map + 1e-4
            acc_probs[:, y:y+win, x:x+win] += prob * attn_map
            acc_weights[y:y+win, x:x+win] += attn_map.squeeze(0)

        acc_weights = acc_weights.clamp(min=1e-6)
        full_probs = acc_probs / acc_weights

        # =========================================================
        #  背景抑制（假设背景是第一个类别）
        # =========================================================
        if classnames[0].lower() in ['background', 'bg']:
            full_probs[0] = full_probs[0] * 0.8   # 降低背景置信度
            # 重新归一化
            full_probs = full_probs / full_probs.sum(dim=0, keepdim=True).clamp(min=1e-6)

        probs_np = full_probs.cpu().numpy()

        # =========================================================
        #  后处理（CRF 等）
        # =========================================================
        if use_crf:
            print(f">>> {os.path.basename(img_path)} 执行 Dense CRF 优化...")
            probs_np = apply_dense_crf(img_np, probs_np)

        max_idx = probs_np.argmax(axis=0)
        masks = torch.stack([torch.from_numpy(max_idx == i) for i in range(C)])

        seg_result = draw_segmentation_masks(img_tensor, masks, colors=custom_palette, alpha=1.0)
        seg_result_pil = TF.to_pil_image(seg_result)
        seg_result_pil.save(out_save_path)
        print(f">>> 保存分割结果: {out_save_path}")

        if show_plot:
            fig, ax = plt.subplots(1, 2, figsize=(20, 10))
            ax[0].imshow(raw_image)
            ax[0].set_title("Original Image")
            ax[0].axis("off")
            ax[1].imshow(seg_result.permute(1, 2, 0).numpy())
            ax[1].set_title("ResQQ + SatUp")
            ax[1].axis("off")
            patches = [mpatches.Patch(color=legend_colors[i], label=classnames[i]) for i in range(C)]
            fig.legend(handles=patches, loc="center right", title="Land Cover Classes")
            plt.subplots_adjust(right=0.88)
            plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="CLIPResQQ 遥感影像批量分割推理")
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", type=str, default=default_device)
    # 单文件模式
    parser.add_argument("--filename", type=str, default=None, help="单张图片路径")
    # 批量文件夹模式
    parser.add_argument("--input_dir", type=str, default=None, help="输入图片文件夹")
    parser.add_argument("--output_dir", type=str, default="./res_output", help="输出结果文件夹")
    # 滑动窗口参数
    parser.add_argument("--window_size", type=int, default=224, help="CLIP窗口大小")
    parser.add_argument("--stride", type=int, default=112, help="滑动步长")
    # 可选开关
    parser.add_argument("--use_crf", action="store_true", help="启用Dense CRF后处理")
    parser.add_argument("--show_plot", action="store_true", help="每张图推理后弹出可视化窗口")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"使用设备: {args.device}")

    # 固定类别与配色
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
        # "playground"
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
        # (250, 200, 20)
    ]

    # 加载模型并固定eval
    model = CLIPResQQ("ViT-B-16", classnames, device=args.device, upsampler="satup" )
    model.eval()

    # 创建输出文件夹
    os.makedirs(args.output_dir, exist_ok=True)

    # 分支1：单文件推理
    if args.filename is not None and os.path.exists(args.filename):
        base_name = os.path.splitext(os.path.basename(args.filename))[0]
        out_path = os.path.join(args.output_dir, f"{base_name}_ours.png")
        infer_single_image(
            img_path=args.filename,
            out_save_path=out_path,
            model=model,
            device=args.device,
            window_size=args.window_size,
            stride=args.stride,
            classnames=classnames,
            custom_palette=custom_palette,
            use_crf=args.use_crf,
            show_plot=args.show_plot
        )

    # 分支2：批量文件夹推理
    elif args.input_dir is not None and os.path.isdir(args.input_dir):
        # 支持常见图片格式
        img_exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG", "*.tif")
        img_paths = []
        for ext in img_exts:
            img_paths.extend(glob.glob(os.path.join(args.input_dir, ext)))
        img_paths = sorted(list(set(img_paths)))

        if len(img_paths) == 0:
            print(f"警告：输入文件夹 {args.input_dir} 未找到任何图片！")
            return

        print(f"共检测到 {len(img_paths)} 张待推理图片")
        for img_path in img_paths:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            out_path = os.path.join(args.output_dir, f"{base_name}_ours.png")
            infer_single_image(
                img_path=img_path,
                out_save_path=out_path,
                model=model,
                device=args.device,
                window_size=args.window_size,
                stride=args.stride,
                classnames=classnames,
                custom_palette=custom_palette,
                use_crf=args.use_crf,
                show_plot=args.show_plot
            )
        print("===== 全部批量推理完成 =====")

    else:
        print("参数错误！请指定 --filename 单图 或 --input_dir 批量文件夹")
        print("示例1(单图): python infer.py --filename test.jpg --show_plot")
        print("示例2(批量): python infer.py --input_dir ./dataset --output_dir ./result --window_size 224 --stride 112")


if __name__ == "__main__":
    main()