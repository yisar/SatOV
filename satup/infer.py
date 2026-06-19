import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

import open_clip
from model import SatUp


# =====================================
# CLIP Feature Extractor
# =====================================
class CLIPViTFeature(nn.Module):
    def __init__(self, device):
        super().__init__()

        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-16",
            pretrained="openai",
        )

        self.visual = model.visual.to(device)
        self.visual.eval()

        for p in self.visual.parameters():
            p.requires_grad = False

        self.embed_dim = 768

    @torch.no_grad()
    def forward(self, x):
        B = x.shape[0]

        x = self.visual.conv1(x)
        H, W = x.shape[-2:]

        x = x.reshape(B, self.embed_dim, -1).permute(0, 2, 1)

        cls = self.visual.class_embedding.to(x.dtype)
        cls = cls + torch.zeros(B, 1, self.embed_dim, device=x.device)

        x = torch.cat([cls, x], dim=1)
        x = x + self.visual.positional_embedding.to(x.dtype)

        x = self.visual.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.visual.transformer(x)
        x = x.permute(1, 0, 2)

        patch = x[:, 1:, :]
        feat = patch.reshape(B, H, W, self.embed_dim)
        feat = feat.permute(0, 3, 1, 2).contiguous()

        return feat


# =====================================
# PCA visualization
# =====================================
def feature_to_rgb(feat):
    C, H, W = feat.shape

    feat = feat.reshape(C, -1).T
    pca = PCA(n_components=3)

    feat = pca.fit_transform(feat)

    feat -= feat.min(0)
    feat /= feat.max(0) + 1e-8

    return feat.reshape(H, W, 3)


# =====================================
def main(args):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)

    ###################################
    # image preprocessing
    ###################################
    image = Image.open(args.image).convert("RGB")

    hr = T.ToTensor()(image).unsqueeze(0)

    # ✔ HR = 224
    hr = F.interpolate(hr, size=(224, 224), mode="bilinear", align_corners=False)

    # ✔ LR = 112
    lr = F.interpolate(hr, scale_factor=0.5, mode="bicubic", align_corners=False)

    hr = hr.to(device)
    lr = lr.to(device)

    ###################################
    # model
    ###################################
    clip_model = CLIPViTFeature(device).to(device)
    model = SatUp(dim=128, v_dim=768).to(device)

    state = torch.load(args.weight, map_location=device)
    model.load_state_dict(state)

    model.eval()

    ###################################
    # inference (✔ fixed alignment)
    ###################################
    with torch.no_grad():
        # ✔ CLIP teacher feature (HR)
        clip_feat = clip_model(hr)

        # ✔ SatUp upsampling: LR → HR feature field
        pred = model(
            image=lr,
            features=clip_feat,
            output_size=(224, 224),
        )

    ###################################
    # visualization (pixel-level feature maps)
    ###################################
    hr_img = hr.squeeze().permute(1, 2, 0).cpu().numpy()
    lr_img = lr.squeeze().permute(1, 2, 0).cpu().numpy()

    clip_vis = feature_to_rgb(clip_feat.squeeze().cpu().numpy())
    pred_vis = feature_to_rgb(pred.squeeze().cpu().numpy())

    ###################################
    # plot
    ###################################
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 2, 1)
    plt.imshow(hr_img)
    plt.title("HR Image (224)")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(lr_img)
    plt.title("LR Image (112)")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(clip_vis)
    plt.title("CLIP Feature (224)")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(pred_vis)
    plt.title("SatUp Upsampled Feature (224)")
    plt.axis("off")

    plt.tight_layout()

    save_path = os.path.join(args.save_dir, "result.png")
    plt.savefig(save_path, dpi=300)
    plt.show()

    print("Saved:", save_path)


# =====================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--image", type=str, default="asset/img2.jpg")
    parser.add_argument("--weight", type=str, default="satup_vitb16_2.pth")
    parser.add_argument("--save_dir", type=str, default="results")

    args = parser.parse_args()
    main(args)
