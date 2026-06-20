import torch
import open_clip
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
from sklearn.decomposition import PCA as SklearnPCA

# =========================
# 1. 辅助类与函数（不变）
# =========================

class UnNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        tensor = tensor.clone()
        if tensor.dim() == 4:
            for t, m, s in zip(tensor, self.mean, self.std):
                t.mul_(s).add_(m)
        else:
            for t, m, s in zip(tensor, self.mean, self.std):
                t.mul_(s).add_(m)
        return tensor


class TorchPCA:
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
    device = image_feats_list[0].device

    def flatten(tensor, target_size=None):
        if target_size is not None and fit_pca is None:
            tensor = torch.nn.functional.interpolate(tensor, (target_size, target_size), mode="area")
        B, C, H, W = tensor.shape
        return tensor.permute(1, 0, 2, 3).reshape(C, B * H * W).permute(1, 0).detach().cpu()

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


def remove_axes(axes):
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
    if not isinstance(hr_or_seg, list):
        hr_or_seg = [hr_or_seg]
    if legend is None:
        legend = ['Image', 'LR Feat'] + [f'HR Feat {i}' for i in range(len(hr_or_seg))]

    feats_list = [lr.unsqueeze(0)] + [h.unsqueeze(0) for h in hr_or_seg]
    reduced, _ = pca(feats_list, dim=3, use_torch_pca=True)

    lr_img = reduced[0].squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    hr_imgs = [r.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() for r in reduced[1:]]

    n_cols = 2 + len(hr_imgs)
    fig, ax = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

    ax[0].imshow(image.permute(1, 2, 0).detach().cpu().numpy())
    ax[0].set_title(legend[0])

    ax[1].imshow(lr_img)
    ax[1].set_title(legend[1])

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
# 2. 修复后的 open_clip Hook 提取器
# =========================

class OpenCLIPFeatureExtractor:
    def __init__(self, model_name="ViT-B-16", pretrained="openai", layer_index=-1, norm=True):
        self.model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        # 提取归一化参数和尺寸
        self.mean, self.std, self.size = self._get_preprocess_info(preprocess)

        self.visual = self.model.visual
        # 获取嵌入维度：从 conv1 输出通道或 positional_embedding 宽度
        self.embed_dim = self.visual.positional_embedding.shape[-1]  # 通常为 768
        # 获取 transformer 块列表（兼容不同属性名）
        if hasattr(self.visual.transformer, 'resblocks'):
            self.blocks = self.visual.transformer.resblocks
        elif hasattr(self.visual.transformer, 'blocks'):
            self.blocks = self.visual.transformer.blocks
        else:
            raise AttributeError("Cannot find transformer blocks in visual.transformer")
        self.num_blocks = len(self.blocks)
        self.layer_index = layer_index
        self.norm = norm
        self._target_layer = self._convert_layer_index(layer_index)
        self._hook_handle = None
        self._hook_output = None

    def _convert_layer_index(self, idx):
        if idx < 0:
            return self.num_blocks + idx
        return min(idx, self.num_blocks - 1)

    def _get_preprocess_info(self, preprocess):
        mean, std, size = None, None, 224
        for t in preprocess.transforms:
            if isinstance(t, T.Normalize):
                mean = t.mean
                std = t.std
            if isinstance(t, T.Resize):
                size = t.size if isinstance(t.size, int) else t.size[0]
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
            std = (0.26862954, 0.26130258, 0.27577711)
        return mean, std, size

    def _hook_fn(self, module, input, output):
        # output 可能是 (N, B, C) 或 (B, N, C)，统一转为 (B, N, C)
        if output.dim() == 3:
            if output.shape[0] != input[0].shape[0]:  # 检查 batch 维度位置
                output = output.permute(1, 0, 2)      # 假设是 (N, B, C) -> (B, N, C)
        self._hook_output = output.detach()

    @torch.no_grad()
    def extract(self, image_tensor):
        # 注册 hook 到目标 resblock
        target_block = self.blocks[self._target_layer]
        self._hook_handle = target_block.register_forward_hook(self._hook_fn)
        # 执行 visual 前向（会经过所有层，但 hook 在目标层触发）
        _ = self.visual(image_tensor)
        self._hook_handle.remove()
        out = self._hook_output  # (B, N, C)
        # 移除 CLS token
        patch_tokens = out[:, 1:, :]  # (B, N-1, C)
        B, N, C = patch_tokens.shape
        H = W = int(np.sqrt(N))
        feats = patch_tokens.reshape(B, H, W, C).permute(0, 3, 1, 2)
        # 可选 LayerNorm（沿通道维）
        if self.norm:
            # 重塑为 (B, H*W, C)
            feats_flat = feats.reshape(B, C, -1).permute(0, 2, 1)
            feats_flat = torch.nn.functional.layer_norm(feats_flat, (C,))
            feats = feats_flat.permute(0, 2, 1).reshape(B, C, H, W)
        return feats


# =========================
# 3. 主程序
# =========================

if __name__ == "__main__":
    model_name = "ViT-B-16"
    pretrained = "openai"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 初始化提取器（最后一层）
    extractor = OpenCLIPFeatureExtractor(model_name, pretrained, layer_index=-1, norm=True)
    extractor.model = extractor.model.to(device)
    mean, std, size = extractor.mean, extractor.std, extractor.size

    # 预处理
    transform = T.Compose([
        T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    img_path = "asset/parrot.png"   # 修改为你的图片路径
    img_pil = Image.open(img_path).convert("RGB")
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    # 提取最后一层特征
    feats_last = extractor.extract(img_tensor)
    print("Last layer shape:", feats_last.shape)  # (1, 768, H, W)

    # 提取倒数第二层（新建提取器）
    extractor2 = OpenCLIPFeatureExtractor(model_name, pretrained, layer_index=-2, norm=True)
    extractor2.model = extractor2.model.to(device)
    feats_second = extractor2.extract(img_tensor)

    # 反归一化原始图像
    unnorm = UnNormalize(mean, std)
    img_display = unnorm(img_tensor.squeeze(0)).clamp(0, 1)

    # 可视化对比
    plot_feats(
        image=img_display,
        lr=feats_last.squeeze(0),
        hr_or_seg=[feats_second.squeeze(0)],
        legend=['Original Image', 'Last Layer', 'Second-Last Layer'],
        save_path='clip_features_openclip_hook.png'
    )