import os
import glob
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
from PIL import Image

import open_clip

from model import SatUp


class Cosine_MSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = torch.nn.MSELoss()
        self.cosine_loss = torch.nn.CosineEmbeddingLoss()

    def forward(self, pred, target):
        pred = rearrange(pred, "b c h w -> (b h w) c")
        target = rearrange(target, "b c h w -> (b h w) c")

        gt = torch.ones_like(target[:, 0])

        # If you must normalize (example: min-max scaling)
        min_val = torch.min(target, dim=1, keepdim=True).values
        max_val = torch.max(target, dim=1, keepdim=True).values
        pred_normalized = (pred - min_val) / (max_val - min_val + 1e-6)
        target_normalized = (target - min_val) / (max_val - min_val + 1e-6)

        return self.cosine_loss(pred, target, gt) + self.mse_loss(
            pred_normalized, target_normalized
        )
# =========================
# Dataset
# =========================
class ImageFolderDataset(Dataset):
    def __init__(self, root, patch_size=224, scale=2):
        self.files = sorted(glob.glob(os.path.join(root, "*.jpg")))
        self.patch_size = patch_size
        self.scale = scale

        self.transform = T.Compose(
            [T.RandomCrop(patch_size), T.RandomHorizontalFlip(), T.ToTensor()]
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        hr = self.transform(img)

        lr = F.interpolate(
            hr.unsqueeze(0),
            scale_factor=1 / self.scale,
            mode="bicubic",
            align_corners=False,
        ).squeeze(0)

        return lr, hr


# =========================
# CLIP Wrapper（修复版）
# =========================
class CLIPViTFeature(nn.Module):
    """
    RADIO-style CLIP wrapper (stable version)
    output: (B, 768, 14, 14)
    """

    def __init__(self, device):
        super().__init__()

        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained="openai"
        )

        self.visual = model.visual.to(device)
        self.visual.eval()

        for p in self.visual.parameters():
            p.requires_grad = False

        self.device = device
        self.embed_dim = 768

    @torch.no_grad()
    def forward(self, x):
        """
        x: (B,3,224,224)
        return: (B,768,14,14)
        """

        B = x.shape[0]

        # patch embedding
        x = self.visual.conv1(x)  # (B,768,14,14)
        H, W = x.shape[-2:]

        x = x.reshape(B, self.embed_dim, -1)  # (B,768,196)
        x = x.permute(0, 2, 1)  # (B,196,768)

        # CLS token
        cls = self.visual.class_embedding.to(x.dtype)
        cls = cls + torch.zeros(B, 1, self.embed_dim, device=x.device)

        x = torch.cat([cls, x], dim=1)  # (B,197,768)

        # pos embedding
        x = x + self.visual.positional_embedding.to(x.dtype)
        x = self.visual.ln_pre(x)

        x = x.permute(1, 0, 2)
        x = self.visual.transformer(x)
        x = x.permute(1, 0, 2)

        # remove CLS
        patch = x[:, 1:, :]  # (B,196,768)

        feat = patch.reshape(B, H, W, self.embed_dim)
        feat = feat.permute(0, 3, 1, 2).contiguous()

        return feat


# =========================
# Train
# =========================
def train(data_root, epochs=50, batch_size=4, lr=2e-4, scale=2):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = ImageFolderDataset(data_root, patch_size=224, scale=scale)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # ⚠️ Windows稳定性修复
        pin_memory=True,
    )

    model = SatUp(dim=128, v_dim=768).to(device)
    clip_feat = CLIPViTFeature(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()

    for epoch in range(epochs):
        total_loss = 0

        for i, (lr_img, hr_img) in enumerate(loader):
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)

            with torch.no_grad():
                features = clip_feat(hr_img)

            pred = model(image=lr_img, features=features, output_size=hr_img.shape[-2:])
            pred = F.adaptive_avg_pool2d(pred, (14, 14))
            # pred = F.adaptive_avg_pool2d(pred, features.shape[-2:])
            loss_fn = Cosine_MSE().to(device)
            loss = loss_fn(pred, features)
            print(i, loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if i % 10 == 0:
                print(f"[E{epoch} | {i}] loss={loss.item():.4f}")

        print(f"Epoch {epoch}: avg loss = {total_loss / len(loader):.4f}")

        torch.save(model.state_dict(), f"satup_vitb16_{epoch}.pth")


# =========================
# main
# =========================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--scale", type=int, default=2)

    args = parser.parse_args()

    train(
        data_root=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        scale=args.scale,
    )
