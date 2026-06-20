import os
import argparse
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import open_clip

from model import SatUp


# =========================
# CLIP (JAFAR aligned: CUT at layer 11 → input of layer 12)
# =========================
class CLIPViTFeature(torch.nn.Module):
    def __init__(self, device):
        super().__init__()

        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained="openai"
        )

        self.visual = model.visual.to(device)
        self.visual.eval()

        for p in self.visual.parameters():
            p.requires_grad = False

        self.embed_dim = 768
        self.device = device

    @torch.no_grad()
    def forward(self, x):
        B = x.shape[0]

        # --------------------
        # patch embedding
        # --------------------
        x = self.visual.conv1(x)  # (B, 768, H/16, W/16)
        H, W = x.shape[-2:]

        x = x.flatten(2).transpose(1, 2)  # (B, N, C)

        # --------------------
        # CLS token
        # --------------------
        cls = self.visual.class_embedding.to(x.dtype)
        cls = cls.unsqueeze(0).unsqueeze(1).expand(B, 1, -1)

        x = torch.cat([cls, x], dim=1)

        # --------------------
        # positional embedding (JAFAR style)
        # --------------------
        pos = self.visual.positional_embedding.to(x.dtype)

        cls_pos = pos[:1]
        patch_pos = pos[1:]

        old_grid = int(patch_pos.shape[0] ** 0.5)

        patch_pos = patch_pos.reshape(1, old_grid, old_grid, -1).permute(0, 3, 1, 2)

        patch_pos = F.interpolate(
            patch_pos, size=(H, W), mode="bicubic", align_corners=False
        )

        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(-1, self.embed_dim)

        pos = torch.cat([cls_pos, patch_pos], dim=0)

        x = x + pos.unsqueeze(0)

        x = self.visual.ln_pre(x)

        # =========================
        # ✔ JAFAR CORE FIX
        # STOP: BEFORE LAYER 12
        # i.e. OUTPUT OF BLOCK 11
        # =========================
        x = x.permute(1, 0, 2)

        for i, blk in enumerate(self.visual.transformer.resblocks):
            if i == 11:  # ⭐第12层输入（0-based）
                break
            x = blk(x)

        x = x.permute(1, 0, 2)

        patch = x[:, 1:, :]

        hw = int(patch.shape[1] ** 0.5)

        feat = patch.reshape(B, hw, hw, self.embed_dim)
        feat = feat.permute(0, 3, 1, 2).contiguous()

        return feat


# =========================
# PCA visualization
# =========================
def feature_to_rgb(feat):
    C, H, W = feat.shape
    feat = feat.reshape(C, -1).T

    pca = PCA(n_components=3)
    feat = pca.fit_transform(feat)

    feat -= feat.min(0)
    feat /= feat.max(0) + 1e-8

    return feat.reshape(H, W, 3)


# =========================
# main
# =========================
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)

    img = Image.open(args.image).convert("RGB")

    hr = T.ToTensor()(img).unsqueeze(0)
    hr = F.interpolate(hr, (224, 224), mode="bilinear")

    lr = F.interpolate(hr, scale_factor=0.5, mode="bicubic")

    hr, lr = hr.to(device), lr.to(device)

    clip = CLIPViTFeature(device)
    model = SatUp(dim=128, v_dim=768).to(device)

    model.load_state_dict(torch.load(args.weight, map_location=device))
    model.eval()

    with torch.no_grad():
        # ✔ GT feature (layer 12 input)
        hr_feat = clip(hr)

        # ✔ LR feature (same extractor)
        lr_feat = clip(lr)

        pred = model(image=lr, features=lr_feat, output_size=hr_feat.shape[-2:])

        pred_vis = F.interpolate(
            pred, size=(224, 224), mode="bilinear", align_corners=False
        )

    hr_img = hr[0].permute(1, 2, 0).cpu().numpy()
    lr_img = lr[0].permute(1, 2, 0).cpu().numpy()

    hr_vis = feature_to_rgb(hr_feat[0].cpu())
    pred_vis = feature_to_rgb(pred_vis[0].cpu())

    plt.figure(figsize=(10, 10))

    plt.subplot(2, 2, 1)
    plt.imshow(hr_img)
    plt.title("HR image")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(lr_img)
    plt.title("LR image")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(hr_vis)
    plt.title("GT feature (Layer 12 input)")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(pred_vis)
    plt.title("Pred feature")
    plt.axis("off")

    plt.tight_layout()
    save_path = os.path.join(args.save_dir, "result.png")
    plt.savefig(save_path, dpi=300)
    plt.show()

    print("saved:", save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="asset/6829.png")
    parser.add_argument("--weight", type=str, default="satup_21.pth")
    parser.add_argument("--save_dir", type=str, default="results")
    args = parser.parse_args()

    main(args)
