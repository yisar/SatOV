import os
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import timm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from model import SatUp


# =========================
# Stable PCA (GT anchor)
# =========================
class FixedPCA:
    def __init__(self, n_components=3):
        self.n_components = n_components

    def fit(self, X):
        self.mean_ = X.mean(dim=0, keepdim=True)
        Xc = X - self.mean_

        U, S, V = torch.pca_lowrank(Xc, q=self.n_components, center=False)
        self.components_ = V[:, :self.n_components]

        # sign fix (very important for stable color)
        for i in range(self.components_.shape[1]):
            if self.components_[:, i].sum() < 0:
                self.components_[:, i] *= -1

        return self

    def transform(self, X):
        return (X - self.mean_) @ self.components_


# =========================
# flatten feature map
# =========================
def flatten_feat(feat):
    B, C, H, W = feat.shape
    return feat.permute(0, 2, 3, 1).reshape(-1, C).detach().cpu()


# =========================
# PCA with GT anchor
# =========================
def stable_pca(gt_feat, pred_feat, lr_feat):
    X = flatten_feat(gt_feat)
    pca = FixedPCA(3).fit(X)

    def apply(feat):
        x = flatten_feat(feat)
        x = pca.transform(x)

        # stable normalization
        x = x - x.min(dim=0, keepdim=True).values
        x = x / (x.max(dim=0, keepdim=True).values + 1e-8)

        B, C, H, W = feat.shape
        return x.reshape(B, H, W, 3).permute(0, 3, 1, 2)

    return apply(gt_feat), apply(pred_feat), apply(lr_feat)


# =========================
# CLIP feature extractor
# =========================
@torch.no_grad()
def extract_clip_feat(model, img):
    result = model.forward_intermediates(
        img,
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


# =========================
# model loader
# =========================
def load_clip():
    model = timm.create_model(
        "vit_base_patch16_clip_384",
        pretrained=True,
        num_classes=0,
        dynamic_img_size=True,
    )
    model.eval()
    return model


# =========================
# main
# =========================
def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    clip = load_clip().to(device)

    config = timm.data.resolve_model_data_config("vit_base_patch16_clip_384")
    mean, std, size = config["mean"], config["std"], config["input_size"][-1]

    # image
    img = Image.open(args.image).convert("RGB")

    hr = T.ToTensor()(img).unsqueeze(0)
    hr = F.interpolate(hr, (size, size), mode="bilinear")

    lr = F.interpolate(hr, scale_factor=0.5, mode="bicubic")

    norm = T.Normalize(mean, std)

    hr_n = norm(hr.clone()).to(device)
    lr_n = norm(lr.clone()).to(device)

    # features
    gt_feat = extract_clip_feat(clip, hr_n)
    lr_feat = extract_clip_feat(clip, lr_n)

    satup = SatUp(dim=256, v_dim=768).to(device)
    satup.load_state_dict(torch.load(args.weight, map_location=device))
    satup.eval()

    with torch.no_grad():
        pred_feat = satup(lr.to(device), lr_feat, (args.output_size, args.output_size))

    # upsample LR for comparison
    lr_feat_up = F.interpolate(lr_feat, (args.output_size, args.output_size), mode="nearest")
    gt_feat_up = F.interpolate(gt_feat, (args.output_size, args.output_size), mode="bilinear")

    # PCA (IMPORTANT FIX)
    gt_pca, pred_pca, lr_pca = stable_pca(gt_feat_up, pred_feat, lr_feat_up)

    # convert to numpy
    img_np = hr[0].permute(1, 2, 0).cpu().numpy()
    gt_np = gt_pca[0].permute(1, 2, 0).cpu().numpy()
    pred_np = pred_pca[0].permute(1, 2, 0).cpu().numpy()
    lr_np = lr_pca[0].permute(1, 2, 0).cpu().numpy()

    # plot
    fig, ax = plt.subplots(1, 4, figsize=(18, 5))

    ax[0].imshow(img_np)
    ax[0].set_title("Image")

    ax[1].imshow(gt_np)
    ax[1].set_title("GT (stable PCA)")

    ax[2].imshow(pred_np)
    ax[2].set_title("SatUp")

    ax[3].imshow(lr_np)
    ax[3].set_title("LR")

    for a in ax:
        a.axis("off")

    plt.tight_layout()

    save_path = Path(args.save_dir) / f"stable_pca_{args.output_size}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print("Saved to:", save_path)


# =========================
# run
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="asset/parrot.png")
    parser.add_argument("--weight", default="satup_49.pth")
    parser.add_argument("--save_dir", default="results")
    parser.add_argument("--output_size", type=int, default=224)

    args = parser.parse_args()
    main(args)