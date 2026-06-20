import torch
import timm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
from sklearn.decomposition import PCA as SklearnPCA

# =========================
# 1. 辅助类与函数
# =========================

class UnNormalize:
    """反归一化，支持单张 (C,H,W) 或批量 (B,C,H,W)"""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        # 复制避免修改原 tensor
        tensor = tensor.clone()
        if tensor.dim() == 4:
            # 批量 (B, C, H, W)
            for t, m, s in zip(tensor, self.mean, self.std):
                t.mul_(s).add_(m)
        else:
            # 单张 (C, H, W)
            for t, m, s in zip(tensor, self.mean, self.std):
                t.mul_(s).add_(m)
        return tensor


class TorchPCA:
    """基于 PyTorch 的 PCA，支持 GPU"""
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        self.mean_ = X.mean(dim=0)
        unbiased = X - self.mean_.unsqueeze(0)
        U, S, V = torch.pca_lowrank(unbiased, q=self.n_components, center=False, niter=4)
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
            tensor = torch.nn.functional.interpolate(tensor, (target_size, target_size), mode="area")
        B, C, H, W = tensor.shape
        return tensor.permute(1, 0, 2, 3).reshape(C, B * H * W).permute(1, 0).detach().cpu()

    # 统一空间尺寸
    if len(image_feats_list) > 1 and fit_pca is None:
        target_size = image_feats_list[0].shape[2]
    else:
        target_size = None

    flattened_feats = []
    for feats in image_feats_list:
        flattened_feats.append(flatten(feats, target_size))
    x = torch.cat(flattened_feats, dim=0)

    # 下采样加速（可选）
    if max_samples is not None and x.shape[0] > max_samples:
        indices = torch.randperm(x.shape[0])[:max_samples]
        x = x[indices]

    if fit_pca is None:
        if use_torch_pca:
            fit_pca = TorchPCA(n_components=dim).fit(x)
        else:
            fit_pca = SklearnPCA(n_components=dim).fit(x)

    reduced_feats = []
    for feats in image_feats_list:
        x_red = fit_pca.transform(flatten(feats))
        if isinstance(x_red, np.ndarray):
            x_red = torch.from_numpy(x_red)
        # 归一化到 [0,1]
        x_red -= x_red.min(dim=0, keepdim=True).values
        x_red /= x_red.max(dim=0, keepdim=True).values + 1e-8
        B, C, H, W = feats.shape
        reduced = x_red.reshape(B, H, W, dim).permute(0, 3, 1, 2).to(device)
        reduced_feats.append(reduced)
    return reduced_feats, fit_pca


def remove_axes(axes):
    """隐藏坐标轴"""
    def _remove(ax):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.xaxis.set_major_formatter(plt.NullFormatter())
        ax.yaxis.set_major_formatter(plt.NullFormatter())

    if hasattr(axes, 'flatten'):
        for ax in axes.flatten():
            _remove(ax)
    else:
        for ax in axes:
            _remove(ax)


def plot_feats(image, lr, hr_or_seg, legend=None, save_path=None):
    """
    显示原图 + 低分辨率特征(PCA) + 高分辨率特征(PCA)
    image: (C, H, W) 反归一化后的图像，范围 [0,1]
    lr: (C, H, W) 低分辨率特征图
    hr_or_seg: list of (C, H, W) 高分辨率特征图
    """
    if not isinstance(hr_or_seg, list):
        hr_or_seg = [hr_or_seg]
    if legend is None:
        legend = ['Image', 'LR Feat'] + [f'HR Feat {i}' for i in range(len(hr_or_seg))]

    # 统一 PCA 降维
    feats_list = [lr.unsqueeze(0)] + [h.unsqueeze(0) for h in hr_or_seg]
    reduced, _ = pca(feats_list, dim=3, use_torch_pca=True)

    lr_img = reduced[0].squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    hr_imgs = [r.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() for r in reduced[1:]]

    n_cols = 2 + len(hr_imgs)
    fig, ax = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

    # 原图
    ax[0].imshow(image.permute(1, 2, 0).detach().cpu().numpy())
    ax[0].set_title(legend[0])

    # LR 特征
    ax[1].imshow(lr_img)
    ax[1].set_title(legend[1])

    # HR 特征
    for idx, hr in enumerate(hr_imgs):
        ax[idx + 2].imshow(hr)
        ax[idx + 2].set_title(legend[idx + 2] if idx + 2 < len(legend) else f'HR {idx}')

    remove_axes(ax)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# =========================
# 2. CLIP 模型加载与特征提取
# =========================

def load_clip_model(model_name="vit_base_patch16_clip_384"):
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


def get_preprocess(model_name="vit_base_patch16_clip_384"):
    data_config = timm.data.resolve_model_data_config(model_name)
    mean = data_config['mean']
    std = data_config['std']
    size = data_config['input_size'][-1]
    return mean, std, size


@torch.no_grad()
def extract_features(model, image_tensor, layer_index=-1, norm=True):
    """
    提取指定层的 patch 特征，返回 (B, C, H, W) 形状
    """
    # 尝试调用 forward_intermediates
    try:
        result = model.forward_intermediates(
            image_tensor,
            indices=[layer_index],
            return_prefix_tokens=True,   # 返回 cls token 和 patch tokens
            norm=norm,
            output_fmt="NCHW",
            intermediates_only=False,
        )
    except TypeError:
        result = model.forward_intermediates(
            image_tensor,
            indices=[layer_index],
            return_prefix_tokens=True,
            output_fmt="NCHW",
            intermediates_only=False,
        )

    # 解析返回值
    if isinstance(result, tuple):
        feats, cls_token = result
    else:
        feats = result
        cls_token = None

    if isinstance(feats, list):
        feats = feats[0]

    # 如果 feats 是 (B, L, C) 序列，则手动重塑为 (B, C, H, W)
    if feats.dim() == 3:
        B, L, C = feats.shape
        # 去掉 cls token（假设 cls 在第一个位置）
        patch_tokens = feats[:, 1:, :]   # (B, L-1, C)
        # 计算 H, W（假设正方形）
        H = W = int((L - 1) ** 0.5)
        # 重塑为 (B, C, H, W)
        feats = patch_tokens.reshape(B, H, W, C).permute(0, 3, 1, 2)
    # 如果已经是 (B, C, H, W) 则直接使用

    return feats


# =========================
# 3. 主程序
# =========================

if __name__ == "__main__":
    model_name = "vit_base_patch16_clip_384"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载模型
    model = load_clip_model(model_name).to(device)
    mean, std, size = get_preprocess(model_name)

    # 图像预处理
    transform = T.Compose([
        T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    # 加载图片（请替换为你的图片路径）
    img_path = "asset/parrot.png"   # <--- 修改此处
    img_pil = Image.open(img_path).convert("RGB")
    img_tensor = transform(img_pil).unsqueeze(0).to(device)   # (1, 3, 384, 384)

    # 提取最后一层特征
    feats_last = extract_features(model, img_tensor, layer_index=-1, norm=True)
    print(f"最后一层特征形状: {feats_last.shape}")   # 应为 (1, 768, 24, 24)

    # 提取倒数第二层特征（用于对比）
    feats_second_last = extract_features(model, img_tensor, layer_index=-2, norm=True)
    print(f"倒数第二层特征形状: {feats_second_last.shape}")

    # 反归一化原始图像
    unnorm = UnNormalize(mean, std)
    img_display = unnorm(img_tensor.squeeze(0))   # (3, 384, 384)
    img_display = torch.clamp(img_display, 0, 1)

    # 可视化对比
    plot_feats(
        image=img_display,
        lr=feats_last.squeeze(0),           # (768, 24, 24)
        hr_or_seg=[feats_second_last.squeeze(0)],
        legend=['Original Image', 'Last Layer', 'Second-Last Layer'],
        save_path='clip_features_comparison.png'
    )