import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torchvision import transforms
from torchvision.utils import draw_segmentation_masks

from model import DenseClip   # 请确保 model.py 中定义了 DenseClip


# ==================== PAMR 模块定义 ====================
class LocalAffinity(nn.Module):
    def __init__(self, dilations=[1]):
        super(LocalAffinity, self).__init__()
        self.dilations = dilations
        weight = self._init_aff()
        self.register_buffer('kernel', weight)

    def _init_aff(self):
        weight = torch.zeros(8, 1, 3, 3)
        for i in range(weight.size(0)):
            weight[i, 0, 1, 1] = 1
        weight[0, 0, 0, 0] = -1
        weight[1, 0, 0, 1] = -1
        weight[2, 0, 0, 2] = -1
        weight[3, 0, 1, 0] = -1
        weight[4, 0, 1, 2] = -1
        weight[5, 0, 2, 0] = -1
        weight[6, 0, 2, 1] = -1
        weight[7, 0, 2, 2] = -1
        return weight

    def forward(self, x):
        B, K, H, W = x.size()
        x = x.view(B*K, 1, H, W)
        x_affs = []
        for d in self.dilations:
            x_pad = F.pad(x, [d]*4, mode='replicate')
            x_aff = F.conv2d(x_pad, self.kernel, dilation=d)
            x_affs.append(x_aff)
        x_aff = torch.cat(x_affs, 1)
        return x_aff.view(B, K, -1, H, W)


class LocalAffinityCopy(LocalAffinity):
    def _init_aff(self):
        weight = torch.zeros(8, 1, 3, 3)
        weight[0, 0, 0, 0] = 1
        weight[1, 0, 0, 1] = 1
        weight[2, 0, 0, 2] = 1
        weight[3, 0, 1, 0] = 1
        weight[4, 0, 1, 2] = 1
        weight[5, 0, 2, 0] = 1
        weight[6, 0, 2, 1] = 1
        weight[7, 0, 2, 2] = 1
        return weight


class LocalStDev(LocalAffinity):
    def _init_aff(self):
        weight = torch.zeros(9, 1, 3, 3)
        weight[0, 0, 0, 0] = 1
        weight[1, 0, 0, 1] = 1
        weight[2, 0, 0, 2] = 1
        weight[3, 0, 1, 0] = 1
        weight[4, 0, 1, 1] = 1
        weight[5, 0, 1, 2] = 1
        weight[6, 0, 2, 0] = 1
        weight[7, 0, 2, 1] = 1
        weight[8, 0, 2, 2] = 1
        return weight

    def forward(self, x):
        x = super(LocalStDev, self).forward(x)
        return x.std(2, keepdim=True)


class LocalAffinityAbs(LocalAffinity):
    def forward(self, x):
        x = super(LocalAffinityAbs, self).forward(x)
        return torch.abs(x)


class PAMR(nn.Module):
    def __init__(self, num_iter=1, dilations=[1]):
        super(PAMR, self).__init__()
        self.num_iter = num_iter
        self.aff_x = LocalAffinityAbs(dilations)
        self.aff_m = LocalAffinityCopy(dilations)
        self.aff_std = LocalStDev(dilations)

    def forward(self, x, mask):
        """
        x: 图像特征，形状 [B, C, H, W]，一般用 RGB 图像（范围 0~1）
        mask: 初始概率图，形状 [B, K, H, W]
        """
        mask = F.interpolate(mask, size=x.size()[-2:], mode="bilinear", align_corners=True)
        B, K, H, W = x.size()
        _, C, _, _ = mask.size()

        x_std = self.aff_std(x)                           # [B,1,H,W]
        x = -self.aff_x(x) / (1e-8 + 0.1 * x_std)         # [B,K,P,H,W]
        x = x.mean(1, keepdim=True)                       # [B,1,P,H,W]
        x = F.softmax(x, 2)                               # 归一化亲和力

        for _ in range(self.num_iter):
            m = self.aff_m(mask)                          # [B,C,P,H,W]
            mask = (m * x).sum(2)                         # [B,C,H,W]
        return mask


# ==================== 辅助函数 ====================
def get_gaussian_mask(size, sigma=0.4):
    """生成中心高、边缘低的高斯权重矩阵"""
    coords = torch.arange(size).float() - (size - 1) / 2
    g = torch.exp(-(coords**2) / (2 * (sigma * size) ** 2))
    mask = g.view(-1, 1) @ g.view(1, -1)
    return mask / mask.max()


def parse_args():
    parser = argparse.ArgumentParser()
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", type=str, default=default_device)
    parser.add_argument("--filename", type=str, default="./asset/img3.jpg")
    parser.add_argument("--window_size", type=int, default=224, help="CLIP 窗口大小")
    parser.add_argument("--stride", type=int, default=112, help="步长，推荐窗口的一半")
    return parser.parse_args()


# ==================== 主函数 ====================
@torch.no_grad()
def main():
    args = parse_args()
    win = args.window_size
    stride = args.stride

    # 类别与配色
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
    model = DenseClip("ViT-B-16", classnames, device=args.device, only_clear=True)
    model.eval()

    clip_norm = transforms.Normalize((0.4814, 0.4578, 0.4082), (0.2686, 0.2613, 0.2757))

    with Image.open(args.filename, "r").convert("RGB") as raw_image:
        w, h = raw_image.size
        img_np = np.array(raw_image)
        img_tensor = TF.to_tensor(raw_image).multiply(255).to(torch.uint8)

        # 初始化累加器
        full_probs = torch.zeros((len(classnames), h, w), device=args.device)
        weight_sum = torch.zeros((1, h, w), device=args.device)
        g_mask = get_gaussian_mask(win).to(args.device)

        # 滑动窗口位置
        y_steps = list(range(0, h - win, stride)) + [h - win]
        x_steps = list(range(0, w - win, stride)) + [w - win]

        print(f">>> 开始滑动窗口推理: {len(y_steps)}x{len(x_steps)} 个切片")
        for y in y_steps:
            for x in x_steps:
                print(f"处理切片: ({x}, {y})")
                crop = raw_image.crop((x, y, x + win, y + win))

                # 预处理
                input_clip = (
                    transforms.Compose([
                        transforms.Resize((win, win)),
                        transforms.ToTensor(),
                        clip_norm,
                    ])(crop)
                    .unsqueeze(0)
                    .to(args.device)
                )

                input_guide = (
                    transforms.Compose([
                        transforms.Resize((win * 2, win * 2)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])(crop)
                    .unsqueeze(0)
                    .to(args.device)
                )

                # 推理
                output = model(input_clip, hr_guide=input_guide)
                output = F.interpolate(output, size=(win, win), mode="bilinear")
                probs = F.softmax(output, dim=1).squeeze(0)

                # 高斯加权融合
                full_probs[:, y:y+win, x:x+win] += probs * g_mask
                weight_sum[:, y:y+win, x:x+win] += g_mask

        # 归一化融合概率
        full_probs /= weight_sum.clamp(min=1e-6)

        # ----- 使用 PAMR 代替 CRF -----
        print(">>> 正在运行 PAMR 精化...")
        # 准备图像特征 (范围 0~1)
        x_img = TF.to_tensor(raw_image).unsqueeze(0).to(args.device)   # [1,3,H,W]
        # 初始概率图加 batch 维度
        mask_init = full_probs.unsqueeze(0)                            # [1,C,H,W]

        pamr = PAMR(num_iter=10, dilations=[1, 2, 4])                  # 多尺度亲和力
        pamr.to(args.device)
        pamr.eval()

        with torch.no_grad():
            refined_probs = pamr(x_img, mask_init)                     # [1,C,H,W]
        refined_probs = refined_probs.squeeze(0).cpu().numpy()         # [C,H,W]

        # 最终类别判定
        max_idx = refined_probs.argmax(axis=0)
        masks = torch.stack([torch.from_numpy(max_idx == i) for i in range(len(classnames))])

        # 渲染分割结果
        seg_result = draw_segmentation_masks(
            img_tensor, masks, colors=custom_palette, alpha=1.0
        )

        # 保存结果
        save_path = f"{args.filename}_pamr.png"
        seg_result_pil = TF.to_pil_image(seg_result)
        seg_result_pil.save(save_path)

        # 可视化
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        ax[0].imshow(raw_image)
        ax[0].set_title("Original Image")
        ax[0].axis("off")

        ax[1].imshow(seg_result.permute(1, 2, 0).numpy())
        ax[1].set_title("Predict Result (PAMR refined)")
        ax[1].axis("off")

        # 添加图例
        patches = [mpatches.Patch(color=legend_colors[i], label=classnames[i]) for i in range(len(classnames))]
        fig.legend(handles=patches, loc="center right", title="Land Cover Classes")
        plt.subplots_adjust(right=0.88)

        plt.savefig(save_path.replace(".png", "_vis.png"), dpi=300, bbox_inches='tight')
        plt.show()

        print(f">>> 处理完成！\n   - 分割结果保存至: {save_path}\n   - 可视化图保存至: {save_path.replace('.png', '_vis.png')}")


if __name__ == "__main__":
    main()