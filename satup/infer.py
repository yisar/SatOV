import os
import argparse
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import timm
from model import SatUp


class TorchPCA:
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
        return (X - self.mean_.unsqueeze(0)) @ self.components_.T


def pca_features(feat_list, dim=3):
    device = feat_list[0].device

    def flatten(t):
        B, C, H, W = t.shape
        return t.permute(1, 0, 2, 3).reshape(C, B * H * W).permute(1, 0).detach().cpu()

    target_size = feat_list[0].shape[2]
    flat_all = []
    for f in feat_list:
        if f.shape[2] != target_size:
            f = F.interpolate(f, (target_size, target_size), mode="area")
        flat_all.append(flatten(f))
    x = torch.cat(flat_all, dim=0)
    pca = TorchPCA(n_components=dim).fit(x)
    reduced = []
    for f in feat_list:
        if f.shape[2] != target_size:
            f = F.interpolate(f, (target_size, target_size), mode="area")
        x_red = pca.transform(flatten(f))
        x_red -= x_red.min(dim=0, keepdim=True).values
        x_red /= x_red.max(dim=0, keepdim=True).values + 1e-8
        B, C, H, W = f.shape
        reduced.append(x_red.reshape(B, H, W, dim).permute(0, 3, 1, 2).to(device))
    return reduced


def load_clip(model_name="vit_base_patch16_clip_384"):
    model = timm.create_model(
        model_name, pretrained=True, num_classes=0, dynamic_img_size=True
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.no_grad()
def extract_clip_feat(model, img_tensor):
    result = model.forward_intermediates(
        img_tensor,
        indices=[-1],
        return_prefix_tokens=True,
        norm=True,
        output_fmt="NCHW",
        intermediates_only=False,
    )
    feats = result[0] if isinstance(result, tuple) else result
    if isinstance(feats, list):
        feats = feats[0]
    if feats.dim() == 3:
        B, L, C = feats.shape
        H = W = int((L - 1) ** 0.5)
        feats = feats[:, 1:].reshape(B, H, W, C).permute(0, 3, 1, 2)
    return feats


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    # CLIP
    clip = load_clip().to(device)
    mean, std, size = (
        timm.data.resolve_model_data_config("vit_base_patch16_clip_384")["mean"],
        timm.data.resolve_model_data_config("vit_base_patch16_clip_384")["std"],
        timm.data.resolve_model_data_config("vit_base_patch16_clip_384")["input_size"][
            -1
        ],
    )

    # Load & preprocess
    img = Image.open(args.image).convert("RGB")
    hr = T.ToTensor()(img).unsqueeze(0)
    hr = F.interpolate(hr, (size, size), mode="bilinear")
    lr = F.interpolate(hr, scale_factor=0.5, mode="bicubic")

    norm = T.Normalize(mean, std)
    hr_n = norm(hr.clone()).to(device)
    lr_n = norm(lr.clone()).to(device)

    # Extract features
    hr_feat = extract_clip_feat(clip, hr_n)  # (1,768,24,24)
    lr_feat = extract_clip_feat(clip, lr_n)  # (1,768,12,12)

    # SatUp
    target = args.output_size
    satup = SatUp(dim=128, v_dim=768).to(device)
    satup.load_state_dict(torch.load(args.weight, map_location=device))
    satup.eval()
    with torch.no_grad():
        pred_feat = satup(lr.to(device), lr_feat, (target, target))

    # Upsample GT and LR features for comparison
    hr_up = F.interpolate(hr_feat, (target, target), mode="bilinear")
    lr_up = F.interpolate(lr_feat, (target, target), mode="nearest")

    # PCA (GT, Pred, LR)
    pcas = pca_features([hr_up, pred_feat, lr_up], dim=3)
    pred_pca = pcas[1].squeeze(0).permute(1, 2, 0).cpu().numpy()
    lr_up_pca = pcas[2].squeeze(0).permute(1, 2, 0).cpu().numpy()

    # Visualize
    hr_disp = hr.squeeze(0).permute(1, 2, 0).cpu().numpy()
    lr_disp = lr.squeeze(0).permute(1, 2, 0).cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0, 0].imshow(hr_disp)
    axes[0, 0].set_title(f"HR {hr.shape[-2]}x{hr.shape[-1]}")
    axes[0, 1].imshow(lr_disp)
    axes[0, 1].set_title(f"LR {lr.shape[-2]}x{lr.shape[-1]}")
    axes[1, 0].imshow(lr_up_pca, interpolation="nearest", cmap='viridis')
    axes[1, 0].set_title(f"LR raw (nearest to {target}x{target})")
    axes[1, 1].imshow(pred_pca, cmap='viridis')
    axes[1, 1].set_title(f"Predicted (SatUp {target}x{target})")
    for ax in axes.flat:
        ax.axis("off")
    plt.tight_layout(pad=0.5)
    plt.savefig(
        os.path.join(args.save_dir, f"result_{target}x{target}.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="asset/img3.jpg")
    parser.add_argument("--weight", default="satup_47.pth")
    parser.add_argument("--save_dir", default="results")
    parser.add_argument("--output_size", type=int, default=224)
    args = parser.parse_args()
    main(args)
