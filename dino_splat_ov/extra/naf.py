#!/usr/bin/env python3
"""
NAF: Zero-Shot Feature Upsampling via Neighborhood Attention Filtering
纯Python可运行版本：支持任意视觉基础模型、任意分辨率的零样本特征上采样
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import PIL
import torch.nn.functional as F
import torchvision.transforms as T
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

# ========== 路径配置：自动识别项目根目录 ==========
project_root = str(Path(__file__).absolute().parent.parent)
sys.path.append(project_root)

from utils.training import load_multiple_backbones
from utils.visualization import plot_feats


def main():
    # ========== 1. 加载Hydra配置 ==========
    GlobalHydra.instance().clear()  # 避免重复运行报错
    initialize(config_path="../config", version_base=None)

    overrides = [
        "val_dataloader.batch_size=1",
        "train_dataloader.batch_size=1",
        "model=naf",
        "img_size=448"
    ]
    cfg = compose(config_name="base", overrides=overrides)

    # ========== 2. 设备与数据加载 ==========
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"运行设备: {device}")

    # 图像预处理：与官方验证集逻辑一致（仅Resize+Crop+ToTensor，归一化后续按模型单独处理）
    IMG_SIZE = 448
    img_transform = T.Compose([
        T.Resize(IMG_SIZE),
        T.CenterCrop(IMG_SIZE),
        T.ToTensor(),
    ])

    # 加载自定义图片
    IMG_PATH = "../asset/dinov3.png"  # 请修改为你的图片路径
    try:
        img_pil = PIL.Image.open(IMG_PATH).convert("RGB")
    except FileNotFoundError:
        print(f"错误：找不到图片 {IMG_PATH}，请修改 IMG_PATH 为正确路径")
        sys.exit(1)

    img_batch = img_transform(img_pil).unsqueeze(0).to(device)

    # ========== 3. 加载NAF预训练上采样模型 ==========
    print("\n正在加载NAF预训练模型...")
    model = torch.hub.load("valeoai/NAF", "naf", pretrained=True, device=device)
    model = model.to(device)
    model.eval()
    print("NAF模型加载完成")

    # ========== 4. 特征上采样+可视化核心函数 ==========
    @torch.no_grad()
    def upsample_backbone(backbone, img, naf_model, mean_std_bck, mean_std_ups, sizes):
        torch.cuda.empty_cache()
        naf_model.eval()
        backbone.eval()

        mean_bck, std_bck = mean_std_bck
        mean_ups, std_ups = mean_std_ups

        # Backbone与NAF模型使用各自的归一化参数
        img_bck = T.functional.normalize(img, mean=mean_bck, std=std_bck)
        img_ups = T.functional.normalize(img, mean=mean_ups, std=std_ups)

        # 提取Backbone原始低分辨率特征
        hr_feats = backbone(img_bck)
        raw_size = hr_feats.shape[-1]

        # NAF上采样到指定分辨率
        preds = []
        for size in sizes:
            pred = naf_model(img_ups, hr_feats, (size, size))
            preds.append(pred)

        # 可视化：原始特征用最近邻上采样作为对比
        hr_feats_vis = F.interpolate(hr_feats, sizes[-1], mode="nearest-exact")
        plot_feats(
            img[0].cpu(),
            hr_feats_vis[0].cpu(),
            [p[0].cpu() for p in preds],
            legend=[f"Input", f"Raw {raw_size}x{raw_size}"] + [f"NAF {s}x{s}" for s in sizes],
            font_size=18,
        )
        plt.tight_layout()
        plt.show()
        torch.cuda.empty_cache()

    # NAF模型固定使用ImageNet标准归一化参数
    mean_ups, std_ups = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    # ========== 示例1：测试多种不同Backbone ==========
    print("\n" + "="*50)
    print("示例1：多视觉基础模型特征上采样演示")
    print("="*50)

    backbone_configs = [
        {"name": "vit_base_patch16_dinov3.lvd1689m"},
        {"name": "radio_v2.5-b"},
        {"name": "franca_vitb14"},
        {"name": "vit_base_patch14_reg4_dinov2"},
        {"name": "vit_base_patch14_dinov2.lvd142m"},
    ]
    name_mapping = {
        "vit_base_patch16_dinov3.lvd1689m": "DINOv3-B",
        "radio_v2.5-b": "Radio-v2.5-B",
        "franca_vitb14": "Franca-B14",
        "vit_base_patch14_reg4_dinov2": "DINOv2-Reg-B",
        "vit_base_patch14_dinov2.lvd142m": "DINOv2-B",
    }

    print("正在加载Backbone模型（首次运行自动下载权重）...")
    backbones, *_ = load_multiple_backbones(cfg, backbone_configs, device)

    for idx, backbone in enumerate(backbones):
        backbone = backbone.to(device).eval()
        bbone_name = name_mapping[backbone_configs[idx]["name"]]
        print(f"\n当前处理: {bbone_name}")

        mean_bck, std_bck = backbone.config["mean"], backbone.config["std"]
        upsample_backbone(
            backbone=backbone,
            img=img_batch,
            naf_model=model,
            mean_std_bck=(mean_bck, std_bck),
            mean_std_ups=(mean_ups, std_ups),
            sizes=[448]
        )

    # ========== 示例2：测试任意上采样分辨率 ==========
    print("\n" + "="*50)
    print("示例2：多分辨率上采样演示（基于DINOv3-B）")
    print("="*50)

    single_config = [{"name": "vit_base_patch16_dinov3.lvd1689m"}]
    single_backbone, *_ = load_multiple_backbones(cfg, single_config, device)
    single_backbone = single_backbone[0].to(device).eval()

    mean_bck, std_bck = single_backbone.config["mean"], single_backbone.config["std"]
    target_sizes = [64, 128, 256, 512, 1024]
    print(f"测试分辨率: {target_sizes}")

    upsample_backbone(
        backbone=single_backbone,
        img=img_batch,
        naf_model=model,
        mean_std_bck=(mean_bck, std_bck),
        mean_std_ups=(mean_ups, std_ups),
        sizes=target_sizes
    )

    print("\n所有演示运行完成！")


if __name__ == "__main__":
    main()