import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch import einsum
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import InterpolationMode
from einops import rearrange
from typing import Tuple
import timm


# ==================== 核心网络模块（保持不变） ====================


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RoPE(nn.Module):
    def __init__(self, dim: int, theta: int = 100):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.freqs = nn.Parameter(torch.empty(2, self.dim))

    def _device_weight_init(self):
        freqs_1d = self.theta ** torch.linspace(0, -1, self.dim // 4)
        freqs_1d = torch.cat([freqs_1d, freqs_1d])
        freqs_2d = torch.zeros(2, self.dim)
        freqs_2d[0, : self.dim // 2] = freqs_1d
        freqs_2d[1, -self.dim // 2 :] = freqs_1d
        self.freqs.data.copy_(freqs_2d * 2 * torch.pi)

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        angle = coords @ self.freqs
        return x * angle.cos() + rotate_half(x) * angle.sin()


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        num_groups=8,
        pad_mode="zeros",
        norm_fn=None,
        activation_fn=nn.SiLU,
        use_conv_shortcut=False,
    ):
        super(ResBlock, self).__init__()
        self.use_conv_shortcut = use_conv_shortcut
        self.norm1 = (
            norm_fn(num_groups, in_channels) if norm_fn is not None else nn.Identity()
        )
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            padding_mode=pad_mode,
            bias=False,
        )
        self.norm2 = (
            norm_fn(num_groups, out_channels) if norm_fn is not None else nn.Identity()
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            padding_mode=pad_mode,
            bias=False,
        )
        self.activation_fn = activation_fn()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                padding=0,
                padding_mode=pad_mode,
                bias=False,
            )

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.activation_fn(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.activation_fn(x)
        x = self.conv2(x)
        if self.use_conv_shortcut or residual.shape != x.shape:
            residual = self.shortcut(residual)
        return x + residual


class SFTModulation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super().__init__()
        self.gamma = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False
        )
        self.beta = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False
        )
        self.norm = nn.GroupNorm(num_groups=8, num_channels=in_channels, affine=False)

    def forward(self, image, features):
        gamma = self.gamma(features)
        beta = self.beta(features)
        return gamma * self.norm(image) + beta


class CrossAttention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, num_heads):
        super().__init__()
        self.norm_q = nn.RMSNorm(query_dim)
        self.norm_k = nn.RMSNorm(key_dim)
        self.norm_v = nn.RMSNorm(value_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=query_dim,
            vdim=value_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )

    def forward(self, query, key, value):
        query = self.norm_q(query)
        key = self.norm_k(key)
        _, attn_scores = self.attention(
            query, key, self.norm_v(value), average_attn_weights=True
        )
        attn_output = einsum("b i j, b j d -> b i d", attn_scores, value)
        return attn_output, attn_scores


class CrossAttentionBlock(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, num_heads, **kwargs):
        super().__init__()
        self.cross_attn = CrossAttention(query_dim, key_dim, value_dim, num_heads)
        self.conv2d = nn.Conv2d(
            query_dim, query_dim, kernel_size=3, stride=1, padding=1, bias=False
        )

    def forward(self, q, k, v, **kwargs):
        q = self.conv2d(q)
        q = rearrange(q, "b c h w -> b (h w) c")
        k = rearrange(k, "b c h w -> b (h w) c")
        v = rearrange(v, "b c h w -> b (h w) c")
        features, _ = self.cross_attn(q, k, v)
        return features


def create_coordinate(h, w, start=0, end=1, device="cuda", dtype=torch.float32):
    x = torch.linspace(start, end, h, device=device, dtype=dtype)
    y = torch.linspace(start, end, w, device=device, dtype=dtype)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    coord_map = torch.stack([xx, yy], axis=-1)[None, ...]
    coords = rearrange(coord_map, "b h w c -> b (h w) c", h=h, w=w)
    return coords


class JAFAR(nn.Module):
    def __init__(
        self, input_dim=3, qk_dim=128, v_dim=384, kernel_size=1, num_heads=4, **kwargs
    ):
        super().__init__()

        def make_encoder(in_dim, kernel_size, num_layers=2):
            return nn.Sequential(
                nn.Conv2d(
                    in_dim,
                    qk_dim,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    padding_mode="reflect",
                    bias=False,
                ),
                *[
                    ResBlock(
                        qk_dim,
                        qk_dim,
                        kernel_size=1,
                        num_groups=8,
                        pad_mode="reflect",
                        norm_fn=nn.GroupNorm,
                        activation_fn=nn.SiLU,
                        use_conv_shortcut=False,
                    )
                    for _ in range(num_layers)
                ],
            )

        self.image_encoder = make_encoder(input_dim, kernel_size=kernel_size)
        self.key_encoder = make_encoder(qk_dim, kernel_size=1)
        self.query_encoder = make_encoder(qk_dim, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=qk_dim, affine=False)
        self.key_features_encoder = make_encoder(v_dim, kernel_size=1)
        self.cross_decode = CrossAttentionBlock(qk_dim, qk_dim, v_dim, num_heads)
        self.sft_key = SFTModulation(qk_dim, qk_dim)
        self.rope = RoPE(qk_dim)
        self.rope._device_weight_init()

    def upsample(self, encoded_image, features, output_size):
        _, _, h, w = features.shape
        queries = self.query_encoder(encoded_image)
        queries = F.adaptive_avg_pool2d(queries, output_size=output_size)
        queries = self.norm(queries)
        keys = self.key_encoder(encoded_image)
        keys = F.adaptive_avg_pool2d(keys, output_size=(h, w))
        keys = self.sft_key(
            keys, self.key_features_encoder(F.normalize(features, dim=1))
        )
        values = features
        out = self.cross_decode(queries, keys, values)
        return out

    def forward(self, image, features, output_size):
        encoded_image = self.image_encoder(image)
        coords = create_coordinate(
            encoded_image.shape[-2], encoded_image.shape[-1], device=image.device
        )
        _, _, h, _ = encoded_image.shape
        encoded_image = rearrange(encoded_image, "b c h w -> b (h w) c")
        encoded_image = self.rope(encoded_image, coords)
        encoded_image = rearrange(encoded_image, "b (h w) c -> b c h w", h=h)
        features = self.upsample(encoded_image, features, output_size)
        features = rearrange(features, "b (h w) c -> b c h w", h=output_size[0])
        return features


class CLIPViTWrapper(nn.Module):
    def __init__(self, name="vit_base_patch16_clip_384", norm=True, **kwargs):
        super().__init__()
        self.name = name
        self.model = timm.create_model(
            name, pretrained=True, num_classes=0, dynamic_img_size=True
        )
        self.model = self.model.eval()
        self.embed_dim = self.model.embed_dim
        self.norm = norm
        self.patch_size = 16
        data_config = timm.data.resolve_model_data_config(model=self.model)
        self.config = data_config

    def forward(self, x: torch.Tensor, n=1) -> Tuple[torch.Tensor, torch.Tensor]:
        feats, cls_token = self.model.forward_intermediates(
            x,
            n,
            return_prefix_tokens=True,
            norm=self.norm,
            output_fmt="NCHW",
            intermediates_only=True,
        )[0]
        return feats, cls_token


# ==================== 损失函数 ====================
class AlignmentLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred = rearrange(pred, "b c h w -> b (h w) c")
        target = rearrange(target, "b c h w -> b (h w) c")
        cos_sim = F.cosine_similarity(pred, target, dim=-1)
        cos_loss = 1 - cos_sim.mean()
        l2_loss = F.mse_loss(pred, target)
        return cos_loss + l2_loss


# ==================== 数据集（仅训练） ====================
class JAFARDataset(Dataset):
    """
    官方训练方式：从高分辨率图像下采样得到低分辨率视图
    """

    def __init__(self, root_dir, hr_size=448, min_scale=2, max_scale=4):
        self.root_dir = root_dir
        self.hr_size = hr_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.image_paths = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.endswith(("png", "jpg", "jpeg"))
        ]

        self.hr_transform = T.Compose(
            [
                T.Resize((hr_size, hr_size), interpolation=InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        from PIL import Image

        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        hr_image = self.hr_transform(image)
        h, w = hr_image.shape[-2:]

        scale = random.uniform(self.min_scale, self.max_scale)
        lr_h, lr_w = int(h / scale), int(w / scale)

        lr_image = F.interpolate(
            hr_image.unsqueeze(0),
            size=(lr_h, lr_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        return {
            "hr_image": hr_image,
            "lr_image": lr_image,
            "guidance": hr_image,
        }


# ==================== 训练函数 ====================
def seed_worker():
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train_one_epoch(
    model, clip_backbone, dataloader, criterion, optimizer, device, epoch, writer
):
    model.train()
    clip_backbone.eval()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        hr_image = batch["hr_image"].to(device)
        lr_image = batch["lr_image"].to(device)
        guidance = batch["guidance"].to(device)

        with torch.no_grad():
            hr_feats, _ = clip_backbone(hr_image)
            lr_feats, _ = clip_backbone(lr_image)

        output_size = hr_feats.shape[-2:]
        pred_feats = model(guidance, lr_feats, output_size)

        loss = criterion(pred_feats, hr_feats)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if batch_idx % 10 == 0:
            print(
                f"Epoch [{epoch}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.6f}"
            )

    avg_loss = total_loss / num_batches
    writer.add_scalar("Train/Loss", avg_loss, epoch)
    return avg_loss


# ==================== 主函数 ====================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 冻结的 CLIP 编码器
    clip_backbone = CLIPViTWrapper(name="vit_base_patch16_clip_384").to(device)
    feature_dim = clip_backbone.embed_dim
    print(f"CLIP feature dim: {feature_dim}")

    # JAFAR 模型
    model = JAFAR(
        input_dim=3, qk_dim=128, v_dim=feature_dim, kernel_size=3, num_heads=4
    ).to(device)

    criterion = AlignmentLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

    # 数据目录（仅训练集）
    data_dir = "data/train"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        print(
            f"Warning: Data directory {data_dir} does not exist. Please put training images there."
        )
        return

    train_dataset = JAFARDataset(data_dir, hr_size=448, min_scale=2, max_scale=4)

    g = torch.Generator()
    g.manual_seed(0)

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        worker_init_fn=seed_worker,
        generator=g,
    )

    log_dir = "runs/jafar_train_only"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    num_epochs = 50

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(
            model,
            clip_backbone,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            writer,
        )
        print(f"Epoch {epoch + 1} Loss: {train_loss:.6f}")

        # 每个epoch保存一次模型（覆盖last）
        torch.save(model.state_dict(), os.path.join(log_dir, "last_model.pth"))
        print(f"Saved model checkpoint at epoch {epoch + 1}")

    writer.close()
    print("Training complete!")


if __name__ == "__main__":
    main()
