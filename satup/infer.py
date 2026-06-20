import os
import argparse
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import timm

# 假设 SatUp 定义在 model.py 中（请确保该文件存在）
from model import SatUp


# =========================
# PCA 辅助类与函数（支持 GPU）
# =========================
class TorchPCA:
    """基于 PyTorch 的 PCA，支持 GPU"""

    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        self.mean_ = X.mean(dim=0)
        unbiased = X - self.mean_.unsqueeze(0)
        U, S, V = torch.pca_lowrank(
            unbiased, q=self.n_components, center=False, niter=4
        )
        self.components_ = V.T
        self.singular_values_ = S
        return self

    def transform(self, X):
        t0 = X - self.mean_.unsqueeze(0)
        projected = t0 @ self.components_.T
        return projected


def pca(image_feats_list, dim=3, fit_pca=None, use_torch_pca=True, max_samples=None):
    """
    对一组特征图进行 PCA 降维，所有特征共享同一个投影空间。
    image_feats_list: list of (B, C, H, W) tensors
    返回降维后的列表，每个形状为 (B, 3, H, W)，值已归一化到 [0,1]
    """
    device = image_feats_list[0].device

    def flatten(tensor, target_size=None):
        if target_size is not None and fit_pca is None:
            tensor = F.interpolate(tensor, (target_size, target_size), mode="area")
        B, C, H, W = tensor.shape
        return (
            tensor.permute(1, 0, 2, 3)
            .reshape(C, B * H * W)
            .permute(1, 0)
            .detach()
            .cpu()
        )

    # 统一空间尺寸（若 fit_pca 为 None，则用第一个特征图的空间尺寸作为目标）
    if len(image_feats_list) > 1 and fit_pca is None:
        target_size = image_feats_list[0].shape[2]
    else:
        target_size = None

    flattened_feats = []
    for feats in image_feats_list:
        flattened_feats.append(flatten(feats, target_size))
    x = torch.cat(flattened_feats, dim=0)

    if max_samples is not None and x.shape[0] > max_samples:
        indices = torch.randperm(x.shape[0])[:max_samples]
        x = x[indices]

    if fit_pca is None:
        if use_torch_pca:
            fit_pca = TorchPCA(n_components=dim).fit(x)
        else:
            from sklearn.decomposition import PCA as SklearnPCA

            fit_pca = SklearnPCA(n_components=dim).fit(x)

    reduced_feats = []
    for feats in image_feats_list:
        x_red = fit_pca.transform(flatten(feats))
        if isinstance(x_red, np.ndarray):
            x_red = torch.from_numpy(x_red)
        x_red -= x_red.min(dim=0, keepdim=True).values
        x_red /= x_red.max(dim=0, keepdim=True).values + 1e-8
        B, C, H, W = feats.shape
        reduced = x_red.reshape(B, H, W, dim).permute(0, 3, 1, 2).to(device)
        reduced_feats.append(reduced)
    return reduced_feats, fit_pca


# =========================
# timm CLIP 特征提取器
# =========================
def load_clip_model(model_name="vit_base_patch16_clip_224"):
    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=0,
        dynamic_img_size=True,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def get_preprocess(model_name="vit_base_patch16_clip_224"):
    data_config = timm.data.resolve_model_data_config(model_name)
    mean = data_config["mean"]
    std = data_config["std"]
    size = data_config["input_size"][-1]
    return mean, std, size


@torch.no_grad()
def extract_features(model, image_tensor, layer_index=10, norm=True):
    """
    提取指定层的 patch 特征，返回 (B, C, H, W) 形状。
    layer_index: 默认为 10，即第 11 个 block 的输出（Layer 12 的输入）
    """
    result = model.forward_intermediates(
        image_tensor,
        indices=[layer_index],
        return_prefix_tokens=True,
        norm=norm,
        output_fmt="NCHW",
        intermediates_only=False,
    )
    if isinstance(result, tuple):
        feats, cls_token = result
    else:
        feats = result
        cls_token = None

    if isinstance(feats, list):
        feats = feats[0]

    # 若返回的是序列 (B, L, C)，去掉 cls 并重塑为 (B, C, H, W)
    if feats.dim() == 3:
        B, L, C = feats.shape
        patch_tokens = feats[:, 1:, :]  # 去掉 cls
        H = W = int((L - 1) ** 0.5)
        feats = patch_tokens.reshape(B, H, W, C).permute(0, 3, 1, 2)
    return feats


# =========================
# 主程序
# =========================
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)

    # 1. 加载 timm CLIP 模型
    model_name = "vit_base_patch16_clip_224"
    clip_model = load_clip_model(model_name).to(device)
    mean, std, _ = get_preprocess(model_name)

    # 2. 读取并预处理图像
    img = Image.open(args.image).convert("RGB")
    # 原始 Tensor (0-1)，用于显示和输入 SatUp（SatUp 期望原始像素值）
    hr_orig = T.ToTensor()(img).unsqueeze(0)  # (1,3,H,W)
    hr_orig = F.interpolate(hr_orig, (224, 224), mode="bilinear")
    lr_orig = F.interpolate(hr_orig, scale_factor=0.5, mode="bicubic")

    # 归一化后输入 CLIP
    normalize = T.Normalize(mean=mean, std=std)
    hr_norm = normalize(hr_orig.clone())
    lr_norm = normalize(lr_orig.clone())
    hr_norm, lr_norm = hr_norm.to(device), lr_norm.to(device)

    # 3. 提取特征 (第 10 层输出)
    hr_feat = extract_features(clip_model, hr_norm, layer_index=10)  # (1, 768, 14, 14)
    lr_feat = extract_features(clip_model, lr_norm, layer_index=10)  # (1, 768, 7, 7)

    print("HR feature shape:", hr_feat.shape)
    print("LR feature shape:", lr_feat.shape)

    # 4. 加载 SatUp 并预测
    model = SatUp(dim=128, v_dim=768).to(device)
    model.load_state_dict(torch.load(args.weight, map_location=device))
    model.eval()

    with torch.no_grad():
        # SatUp 输入为 LR 原始图像（0-1）和 LR 特征，输出目标尺寸的特征
        pred = model(
            image=lr_orig.to(device), features=lr_feat, output_size=hr_feat.shape[-2:]
        )
        # 将预测特征上采样到与 HR 特征相同尺寸（用于可视化对比，若原始输出尺寸即匹配则无需插值）
        pred_up = F.interpolate(
            pred, size=hr_feat.shape[-2:], mode="bilinear", align_corners=False
        )

    # 5. 生成对比所需的数据
    # 5a. LR 特征双线性上采样（简单插值基线）
    lr_feat_up = F.interpolate(lr_feat, size=hr_feat.shape[-2:], mode="bilinear")

    # 5b. 共享 PCA 降维（GT, Pred, LR_up）
    feats_for_pca = [hr_feat, pred_up, lr_feat_up]  # 三者尺寸均为 (1,768,14,14)
    reduced, _ = pca(feats_for_pca, dim=3, use_torch_pca=True)
    hr_pca = reduced[0].squeeze(0).permute(1, 2, 0).cpu().numpy()
    pred_pca = reduced[1].squeeze(0).permute(1, 2, 0).cpu().numpy()
    lr_up_pca = reduced[2].squeeze(0).permute(1, 2, 0).cpu().numpy()

    # 5c. 图像显示准备
    hr_display = hr_orig.squeeze(0).permute(1, 2, 0).cpu().numpy()
    lr_display = lr_orig.squeeze(0).permute(1, 2, 0).cpu().numpy()
    lr_bicubic = F.interpolate(lr_orig, size=hr_orig.shape[-2:], mode="bicubic")
    lr_bicubic_display = lr_bicubic.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # 6. 2行×3列 可视化布局
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 第一行：图像
    axes[0, 0].imshow(hr_display)
    axes[0, 0].set_title("HR image (224×224)")
    axes[0, 1].imshow(lr_bicubic_display)
    axes[0, 1].set_title("LR bicubic upsampled")
    axes[0, 2].imshow(lr_display)
    axes[0, 2].set_title("LR original (112×112)")

    # 第二行：特征（PCA 伪彩色）
    axes[1, 0].imshow(hr_pca)
    axes[1, 0].set_title("GT feature")
    axes[1, 1].imshow(pred_pca)
    axes[1, 1].set_title("Pred feature (SatUp)")
    axes[1, 2].imshow(lr_up_pca)
    axes[1, 2].set_title("LR feat up (bilinear)")

    for ax in axes.flat:
        ax.axis("off")
    plt.tight_layout()
    save_path = os.path.join(args.save_dir, "result_timm_upsample.png")
    plt.savefig(save_path, dpi=300)
    plt.show()
    print("saved:", save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="asset/parrot.png")
    parser.add_argument("--weight", type=str, default="satup_12.pth")
    parser.add_argument("--save_dir", type=str, default="results")
    args = parser.parse_args()
    main(args)
