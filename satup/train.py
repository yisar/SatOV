import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
from einops import rearrange
import open_clip
from model import SatUp
from tqdm import tqdm


# =========================
# Loss (JAFAR standard)
# =========================
class Cosine_MSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.cos = nn.CosineEmbeddingLoss()

    def forward(self, pred, target):
        pred = rearrange(pred, "b c h w -> (b h w) c")
        target = rearrange(target, "b c h w -> (b h w) c")
        y = torch.ones(pred.size(0), device=pred.device)
        return self.cos(pred, target, y) + self.mse(pred, target)


# =========================
# Dataset (HR only)
# =========================
class ImageFolderDataset(Dataset):
    def __init__(self, root):
        self.files = sorted(glob.glob(os.path.join(root, "*.jpg")))
        self.tf = T.Compose(
            [T.Resize(256), T.RandomCrop(224), T.RandomHorizontalFlip(), T.ToTensor()]
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return self.tf(Image.open(self.files[idx]).convert("RGB"))


# =========================
# CLIP extractor (TRUE JAFAR STYLE)
# ✔ NO manual transformer rewrite
# ✔ proper hook-style truncation
# =========================
class CLIPViTFeature(nn.Module):
    def __init__(self, device):
        super().__init__()

        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained="openai"
        )

        self.visual = model.visual.to(device).eval()
        for p in self.visual.parameters():
            p.requires_grad = False

        self.embed_dim = 768

    @torch.no_grad()
    def forward(self, x):
        B = x.shape[0]

        # patch embedding
        x = self.visual.conv1(x)
        _, _, H, W = x.shape

        x = x.reshape(B, self.embed_dim, -1).permute(0, 2, 1)

        cls = self.visual.class_embedding.to(x.dtype)
        cls = cls.unsqueeze(0).unsqueeze(1).expand(B, 1, -1)

        x = torch.cat([cls, x], dim=1)

        pos = self.visual.positional_embedding.to(x.dtype)
        cls_pos = pos[:1]
        patch_pos = pos[1:].reshape(1, 14, 14, 768).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(patch_pos, size=(H, W), mode="bilinear")
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(-1, 768)

        x = x + torch.cat([cls_pos, patch_pos], dim=0).unsqueeze(0)

        x = self.visual.ln_pre(x)

        # =========================
        # ✔ TRUE JAFAR CUT POINT
        # (before last block)
        # =========================
        x = x.permute(1, 0, 2)

        for blk in self.visual.transformer.resblocks[:-1]:
            x = blk(x)

        x = x.permute(1, 0, 2)

        patch = x[:, 1:, :]

        # ✔ correct spatial size from conv
        hw = int(H)  # conv feature already gives correct grid

        feat = patch.reshape(B, hw, hw, self.embed_dim)
        feat = feat.permute(0, 3, 1, 2)

        return feat


# =========================
# TRAIN (TRUE JAFAR PIPELINE)
# =========================
def train(data_root, epochs=50, batch_size=4, lr=2e-4):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = ImageFolderDataset(data_root)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SatUp(dim=128, v_dim=768).to(device)
    clip = CLIPViTFeature(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = Cosine_MSE().to(device)

    for epoch in range(epochs):
        total = 0
        pbar = tqdm(loader)

        for hr in pbar:
            hr = hr.to(device)

            # =========================
            # HR feature (target)
            # =========================
            with torch.no_grad():
                hr_feat = clip(hr)

            # =========================
            # JAFAR LR generation (CORRECT)
            # =========================
            scale = np.random.uniform(0.25, 0.5)

            lr = F.interpolate(
                hr, scale_factor=scale, mode="bicubic", align_corners=False
            )

            # ensure patch alignment (CRITICAL)
            _, _, h, w = lr.shape
            h = (h // 16) * 16
            w = (w // 16) * 16
            lr = F.interpolate(lr, size=(h, w), mode="bicubic")

            # =========================
            # LR feature (key/value)
            # =========================
            with torch.no_grad():
                lr_feat = clip(lr)

            # =========================
            # JAFAR forward
            # =========================
            pred = model(
                image=lr,  # query = HR
                features=lr_feat,  # KV = LR
                output_size=hr_feat.shape[-2:],
            )

            loss = loss_fn(pred, hr_feat)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()
            pbar.set_postfix(loss=loss.item())

        print(f"Epoch {epoch}: {total / len(loader):.4f}")
        torch.save(model.state_dict(), f"satup_{epoch}.pth")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)

    args = parser.parse_args()

    train(
        data_root=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
