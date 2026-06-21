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
from tqdm import tqdm

from model import SatUp


# =========================
# 防止超大图报错（遥感数据必须加）
# =========================
Image.MAX_IMAGE_PIXELS = None


# =========================
# ClearCLIP Feature Extractor
# =========================
class ClearCLIPFeature(nn.Module):
    def __init__(self, model_name="ViT-B-16", device="cuda"):
        super().__init__()
        self.device = device

        model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained="openai", device=device
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
                patch_pos, size=(grid_h, grid_w), mode="bicubic", align_corners=False
            )
            patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, grid_h * grid_w, -1)
            pos_embed = torch.cat([cls_pos, patch_pos.squeeze(0)], dim=0)

        x_tokens = torch.cat([cls_token.unsqueeze(0).expand(B, 1, -1), x_tokens], dim=1)

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

        attn_map = (q @ q.transpose(-2, -1)) * (dim_head**-0.5)
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
# Loss
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
# Dataset
# =========================
class ImageFolderDataset(Dataset):
    def __init__(self, root, mean, std, size=384):
        self.files = sorted(glob.glob(os.path.join(root, "*.jpg")))

        self.tf = T.Compose(
            [
                T.Resize((size, size)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return self.tf(Image.open(self.files[idx]).convert("RGB"))


# =========================
# TRAIN
# =========================
def train(data_root, epochs=50, batch_size=4, lr=2e-4, model_name="ViT-B-16"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    extractor = ClearCLIPFeature(model_name, device)

    # =========================
    # FIX: open_clip 返回值是3个
    # =========================
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained="openai", device=device
    )

    mean = preprocess.transforms[-1].mean
    std = preprocess.transforms[-1].std

    dataset = ImageFolderDataset(data_root, mean, std)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model = SatUp(dim=128, v_dim=extractor.embed_dim).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = Cosine_MSE().to(device)

    for epoch in range(epochs):
        model.train()
        total = 0
        pbar = tqdm(loader)

        for hr in pbar:
            hr = hr.to(device)

            with torch.no_grad():
                hr_feat = extractor(hr)

            query_img = F.interpolate(
                hr, scale_factor=0.5, mode="bicubic", align_corners=False
            )

            scale = np.random.uniform(0.25, 0.5)

            kv_img = F.interpolate(
                hr, scale_factor=scale, mode="bicubic", align_corners=False
            )

            with torch.no_grad():
                lr_feat = extractor(kv_img)

            pred = model(
                image=query_img, features=lr_feat, output_size=hr_feat.shape[-2:]
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
    parser.add_argument("--model_name", type=str, default="ViT-B-16")

    args = parser.parse_args()

    train(
        data_root=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        model_name=args.model_name,
    )
