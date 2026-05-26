import argparse
import os
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

from model import DenseClip


def get_gaussian_mask(size, sigma=0.4):
    """生成中心权重高、边缘权重低的高斯矩阵，用于平滑接缝"""
    coords = torch.arange(size).float() - (size - 1) / 2
    g = torch.exp(-(coords**2) / (2 * (sigma * size) ** 2))
    mask = g.view(-1, 1) @ g.view(1, -1)
    return mask / mask.max()


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


def parse_args():
    parser = argparse.ArgumentParser()
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", type=str, default=default_device)
    parser.add_argument("--filename", type=str, default="./asset/img3.jpg")
    parser.add_argument("--window_size", type=int, default=224, help="CLIP 窗口大小")
    parser.add_argument(
        "--stride", type=int, default=112, help="步长，推荐窗口的一半实现重叠"
    )
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    win = args.window_size
    stride = args.stride

    # 类别定义与配色
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

    # 加载模型
    model = DenseClip("ViT-B-16", classnames, device=args.device)
    model.eval()

    clip_norm = transforms.Normalize((0.4814, 0.4578, 0.4082), (0.2686, 0.2613, 0.2757))

    with Image.open(args.filename, "r").convert("RGB") as raw_image:
        w, h = raw_image.size
        img_np = np.array(raw_image)
        img_tensor = TF.to_tensor(raw_image).multiply(255).to(torch.uint8)

        # 初始化全局累加器
        full_probs = torch.zeros((len(classnames), h, w), device=args.device)
        weight_sum = torch.zeros((1, h, w), device=args.device)
        g_mask = get_gaussian_mask(win).to(args.device)

        # 采样点逻辑
        y_steps = list(range(0, h - win, stride)) + [h - win]
        x_steps = list(range(0, w - win, stride)) + [w - win]

        print(f">>> 开始滑动窗口推理: {len(y_steps)}x{len(x_steps)} 个切片")
        for y in y_steps:
            for x in x_steps:
                print(x, y)
                crop = raw_image.crop((x, y, x + win, y + win))

                # 预处理
                input_clip = (
                    transforms.Compose(
                        [
                            transforms.Resize((win, win)),
                            transforms.ToTensor(),
                            clip_norm,
                        ]
                    )(crop)
                    .unsqueeze(0)
                    .to(args.device)
                )

                input_guide = (
                    transforms.Compose(
                        [
                            transforms.Resize((win * 2, win * 2)),
                            transforms.ToTensor(),
                            transforms.Normalize(
                                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                            ),
                        ]
                    )(crop)
                    .unsqueeze(0)
                    .to(args.device)
                )

                # 推理
                output = model(input_clip, hr_guide=input_guide)
                output = F.interpolate(output, size=(win, win), mode="bilinear")
                probs = F.softmax(output, dim=1).squeeze(0)

                # 高斯加权融合
                full_probs[:, y : y + win, x : x + win] += probs * g_mask
                weight_sum[:, y : y + win, x : x + win] += g_mask

        # 1. 归一化融合后的概率图
        full_probs /= weight_sum.clamp(min=1e-6)
        probs_np = full_probs.cpu().numpy()

        # 2. 执行全局 Dense CRF 优化 (核心步骤)
        print(">>> 正在运行全局 Dense CRF 优化，请稍候...")
        refined_probs = apply_dense_crf(img_np, probs_np)

        # 3. 最终类别判定
        max_idx = refined_probs.argmax(axis=0)
        masks = torch.stack(
            [torch.from_numpy(max_idx == i) for i in range(len(classnames))]
        )

        # 4. 渲染不透明掩码 (alpha=1.0)
        seg_result = draw_segmentation_masks(
            img_tensor, masks, colors=custom_palette, alpha=1.0
        )
        save_path = f"{args.filename}"
        # seg_result_pil = TF.to_pil_image(seg_result)
        # seg_result_pil.save(save_path.replace("dataset", "res"))

        # 5. 可视化
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        ax[0].imshow(raw_image)
        ax[0].set_title("Original Image")
        ax[0].axis("off")

        ax[1].imshow(seg_result.permute(1, 2, 0).numpy())
        ax[1].set_title("Predict Result")
        ax[1].axis("off")

        # 添加图例
        patches = [
            mpatches.Patch(color=legend_colors[i], label=classnames[i])
            for i in range(len(classnames))
        ]
        fig.legend(handles=patches, loc="center right", title="Land Cover Classes")
        plt.subplots_adjust(right=0.88)

        # save_path = f'{args.filename}'
        # plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f">>> 处理完成！高清结果已保存至: {save_path}")


if __name__ == "__main__":
    main()
