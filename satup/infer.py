import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import open_clip
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

        # sign fix for stability
        for i in range(self.components_.shape[1]):
            if self.components_[:, i].sum() < 0:
                self.components_[:, i] *= -1

        return self

    def transform(self, X):
        return (X - self.mean_) @ self.components_


def flatten_feat(feat):
    B, C, H, W = feat.shape
    return feat.permute(0, 2, 3, 1).reshape(-1, C).detach().cpu()


def stable_pca(gt_feat, pred_feat, lr_feat):
    X = flatten_feat(gt_feat)
    pca = FixedPCA(3).fit(X)

    def apply(feat):
        x = flatten_feat(feat)
        x = pca.transform(x)

        x = x - x.min(dim=0, keepdim=True).values
        x = x / (x.max(dim=0, keepdim=True).values + 1e-8)

        B, C, H, W = feat.shape
        return x.reshape(B, H, W, 3).permute(0, 3, 1, 2)

    return apply(gt_feat), apply(pred_feat), apply(lr_feat)


# =========================
# ClearCLIP Feature Extractor
# =========================
class ClearCLIPFeature(nn.Module):
    def __init__(self, model_name="ViT-B-16", device="cuda"):
        super().__init__()
        self.device = device

        model, _, _ = open_clip.create_model_and_transforms(
            model_name,
            pretrained="openai",
            device=device
        )

        self.visual = model.visual.eval()

        for p in self.visual.parameters():
            p.requires_grad = False

        self.embed_dim = self.visual.conv1.out_channels

    @torch.no_grad()
    def forward(self, x):
        B, C, H, W = x.shape

        x = self.visual.conv1(x)
        grid_h, grid_w = x.shape[-2:]

        x_tokens = x.flatten(2).transpose(1, 2)

        cls_token = self.visual.class_embedding.to(x_tokens.dtype)
        pos_embed = self.visual.positional_embedding.to(x_tokens.dtype)

        cls_pos = pos_embed[:1]
        patch_pos = pos_embed[1:]

        old_grid = int(patch_pos.shape[0] ** 0.5)

        if old_grid != grid_h or old_grid != grid_w:
            patch_pos = patch_pos.reshape(1, old_grid, old_grid, -1).permute(0, 3, 1, 2)
            patch_pos = F.interpolate(
                patch_pos, size=(grid_h, grid_w),
                mode="bicubic",
                align_corners=False
            )
            patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, grid_h * grid_w, -1)
            pos_embed = torch.cat([cls_pos, patch_pos.squeeze(0)], dim=0)

        x_tokens = torch.cat(
            [cls_token.unsqueeze(0).expand(B, 1, -1), x_tokens],
            dim=1
        )

        x_tokens = x_tokens + pos_embed
        x_tokens = self.visual.ln_pre(x_tokens)

        blocks = self.visual.transformer.resblocks
        for i in range(len(blocks) - 1):
            x_tokens = blocks[i](x_tokens)

        last = blocks[-1]

        x = last.ln_1(x_tokens)
        attn = last.attn

        qkv = F.linear(x, attn.in_proj_weight, attn.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)

        B, N, D = q.shape
        heads = attn.num_heads
        dim_head = D // heads

        q = q.view(B, N, heads, dim_head).transpose(1, 2)
        v = v.view(B, N, heads, dim_head).transpose(1, 2)

        attn_map = (q @ q.transpose(-2, -1)) * (dim_head ** -0.5)
        attn_map = attn_map.softmax(dim=-1)

        out = (attn_map @ v).transpose(1, 2).reshape(B, N, D)

        out = F.linear(out, attn.out_proj.weight, attn.out_proj.bias)
        out = self.visual.ln_post(out)

        cls = out[:, :1]
        patch = out[:, 1:]
        patch = patch - cls

        feat = patch.permute(0, 2, 1).reshape(B, D, grid_h, grid_w)

        return feat


# =========================
# main
# =========================
def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    clip_encoder = ClearCLIPFeature(
        model_name="ViT-B-16",
        device=device
    ).to(device)

    clip_encoder.eval()

    # image
    img = Image.open(args.image).convert("RGB")

    hr = T.ToTensor()(img).unsqueeze(0).to(device)
    hr = F.interpolate(hr, (args.output_size, args.output_size), mode="bilinear")

    lr = F.interpolate(hr, scale_factor=0.5, mode="bicubic")

    # features (ALL via ClearCLIP)
    gt_feat = clip_encoder(hr)
    lr_feat = clip_encoder(lr)

    satup = SatUp(dim=128, v_dim=768).to(device)
    satup.load_state_dict(torch.load(args.weight, map_location=device, weights_only=True))
    satup.eval()

    with torch.no_grad():
        pred_feat = satup(
            lr,
            lr_feat,
            (args.output_size, args.output_size)
        )

    # upsample for comparison
    lr_feat_up = F.interpolate(lr_feat, (args.output_size, args.output_size), mode="nearest")
    gt_feat_up = F.interpolate(gt_feat, (args.output_size, args.output_size), mode="bilinear")

    # PCA
    gt_pca, pred_pca, lr_pca = stable_pca(gt_feat_up, pred_feat, lr_feat_up)

    # to numpy
    img_np = hr[0].permute(1, 2, 0).detach().cpu().numpy()
    gt_np = gt_pca[0].permute(1, 2, 0).detach().cpu().numpy()
    pred_np = pred_pca[0].permute(1, 2, 0).detach().cpu().numpy()
    lr_np = lr_pca[0].permute(1, 2, 0).detach().cpu().numpy()

    # plot
    fig, ax = plt.subplots(1, 4, figsize=(18, 5))

    ax[0].imshow(img_np)
    ax[0].set_title("Image")

    ax[1].imshow(lr_np)
    ax[1].set_title("LR")

    ax[2].imshow(gt_np)
    ax[2].set_title("GT")

    ax[3].imshow(pred_np)
    ax[3].set_title("SatUp")


    for a in ax:
        a.axis("off")

    plt.tight_layout()

    save_path = Path(args.save_dir) / f"clearclip_pca_{args.output_size}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print("Saved to:", save_path)


# =========================
# run
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="asset/P0016654.jpg")
    parser.add_argument("--weight", default="satup_a_15.pth")
    parser.add_argument("--save_dir", default="results")
    parser.add_argument("--output_size", type=int, default=224)

    args = parser.parse_args()
    main(args)