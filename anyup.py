import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


# ---------------------------- Helper functions ----------------------------
def reflect_pad(x, padding):
    """Reflect-pad spatial dims of NHWC tensor."""
    if padding <= 0:
        return x
    # PyTorch's pad expects (left, right, top, bottom)
    return F.pad(
        x.permute(0, 3, 1, 2), (padding, padding, padding, padding), mode="reflect"
    ).permute(0, 2, 3, 1)


def pool_to(x, size):
    """Area pool: NHWC -> NHWC, downsampling to given spatial size."""
    b, h, w, c = x.shape
    oh, ow = size
    if h == oh and w == ow:
        return x
    return x.reshape(b, oh, h // oh, ow, w // ow, c).mean(dim=(2, 4))


def create_coordinate(h, w, device=None):
    """Return normalized coordinates (1, H*W, 2) in [0,1] range."""
    x = torch.linspace(0, 1, h, device=device)
    y = torch.linspace(0, 1, w, device=device)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    coords = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
    return coords.unsqueeze(0)  # (1, H*W, 2)


# ---------------------------- ResBlock ----------------------------
class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        num_groups=8,
        use_norm=True,
        use_conv_shortcut=False,
    ):
        super().__init__()
        p = kernel_size // 2
        layers = []
        if use_norm:
            layers.append(nn.GroupNorm(num_groups, in_channels))
        layers.append(nn.SiLU())
        layers.append(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=p, bias=False)
        )
        if use_norm:
            layers.append(nn.GroupNorm(num_groups, out_channels))
        layers.append(nn.SiLU())
        layers.append(
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=p, bias=False)
        )
        self.block = nn.Sequential(*layers)
        if use_conv_shortcut or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        return self.block(x) + self.shortcut(x)


# ---------------------------- LearnedFeatureUnification ----------------------------
class LearnedFeatureUnification(nn.Module):
    """Depthwise convolution with channel-wise softmax & averaging."""

    def __init__(self, out_channels, kernel_size=3):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        # basis shape: (out_channels, kernel_size, kernel_size, 1)
        self.basis = nn.Parameter(
            torch.randn(out_channels, kernel_size, kernel_size, 1) * 0.01
        )

    def forward(self, features):
        b, h, w, c = features.shape
        k = self.kernel_size
        p = k // 2
        # Pad spatially
        x = F.pad(features.permute(0, 3, 1, 2), (p, p, p, p), mode="reflect").permute(
            0, 2, 3, 1
        )

        parts = []
        for ci in range(c):
            inp = x[..., ci : ci + 1]  # (b, h, w, 1)
            weight = self.basis.unsqueeze(1)  # (out_channels, 1, k, k)
            out = F.conv2d(inp.permute(0, 3, 1, 2), weight, padding=0).permute(
                0, 2, 3, 1
            )
            parts.append(out)
        x_flat = torch.cat(parts, dim=-1)  # (b, h, w, out_channels * c)

        # Border normalization
        mask = torch.ones(1, h, w, 1, device=x.device)
        mask_padded = F.pad(
            mask.permute(0, 3, 1, 2), (p, p, p, p), mode="constant", value=0
        ).permute(0, 2, 3, 1)
        ones_kern = torch.ones(1, k, k, 1, device=x.device)
        denom = F.conv2d(
            mask_padded.permute(0, 3, 1, 2), ones_kern.permute(0, 3, 1, 2), padding=0
        ).permute(0, 2, 3, 1)
        x_flat = x_flat / (denom + 1e-6)

        x_5d = x_flat.view(b, h, w, self.out_channels, c)
        attn = F.softmax(x_5d, dim=3)
        return attn.mean(dim=4)  # (b, h, w, out_channels)


# ---------------------------- Axis-Aligned Sinusoidal Embedding (Official) ----------------------------
class AnyUpPositionalEmbedding(nn.Module):
    """Axis-aligned Sinusoidal Embedding for 2D positions.
    Mimics the effect of a 2D RoPE but with independent sinusoids per dimension."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # Learnable frequencies for row and column directions
        self.freqs_r = nn.Parameter(torch.randn(1, 1, dim // 2) * 0.02)
        self.freqs_c = nn.Parameter(torch.randn(1, 1, dim // 2) * 0.02)

    def _rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, x, coords):
        B, N, D = x.shape
        # coords: (B, N, 2), normalized coordinates in [0, 1]
        r = coords[..., 0:1]  # (B, N, 1)
        c = coords[..., 1:2]  # (B, N, 1)

        # Compute angles: angle = position * frequency
        angle_r = 2 * torch.pi * r * self.freqs_r  # (B, N, D//2)
        angle_c = 2 * torch.pi * c * self.freqs_c  # (B, N, D//2)
        angle = torch.cat([angle_r, angle_c], dim=-1)  # (B, N, D)

        cos = torch.cos(angle)
        sin = torch.sin(angle)
        rotated = self._rotate_half(x)
        return x * cos + rotated * sin


# ---------------------------- Window mask ----------------------------
def tile_mask(rs, re, w_out, h_out, krs, kre, w_kv, h_kv, ratio, device):
    tile_h = re - rs
    kvt_h = kre - krs
    q_r = (torch.arange(rs, re, device=device).float() + 0.5) / h_out
    q_c = (torch.arange(w_out, device=device).float() + 0.5) / w_out
    k_r_idx = torch.arange(krs, kre, device=device)
    k_c_idx = torch.arange(w_kv, device=device)
    qr_r0 = torch.floor(torch.clamp(q_r - ratio, 0, 1) * h_kv).long()
    qr_r1 = torch.ceil(torch.clamp(q_r + ratio, 0, 1) * h_kv).long()
    r_ok = (k_r_idx[None, :] >= qr_r0[:, None]) & (k_r_idx[None, :] < qr_r1[:, None])
    qc_c0 = torch.floor(torch.clamp(q_c - ratio, 0, 1) * w_kv).long()
    qc_c1 = torch.ceil(torch.clamp(q_c + ratio, 0, 1) * w_kv).long()
    c_ok = (k_c_idx[None, :] >= qc_c0[:, None]) & (k_c_idx[None, :] < qc_c1[:, None])
    mask = (r_ok[:, None, :, None] & c_ok[None, :, None, :]).reshape(
        tile_h * w_out, kvt_h * w_kv
    )
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, Q, KV)


# ---------------------------- Cross attention modules ----------------------------
class FlexCrossAttention(nn.Module):
    def __init__(self, qk_dim, num_heads):
        super().__init__()
        self.dim = qk_dim
        self.num_heads = num_heads
        self.norm_q = nn.LayerNorm(qk_dim)
        self.norm_k = nn.LayerNorm(qk_dim)
        self.q_proj = nn.Linear(qk_dim, qk_dim, bias=False)
        self.k_proj = nn.Linear(qk_dim, qk_dim, bias=False)
        self.v_proj = nn.Linear(qk_dim, qk_dim, bias=False)
        self.out_proj = nn.Linear(qk_dim, qk_dim, bias=False)

    def forward(self, query, key, value, mask=None):
        b, nq, d = query.shape
        nk = key.shape[1]

        q = self.norm_q(query)
        k = self.norm_k(key)
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(value)

        head_dim = d // self.num_heads
        q = q.view(b, nq, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(b, nk, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(b, nk, self.num_heads, head_dim).transpose(1, 2)

        scale = head_dim**-0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        if mask is not None:
            # mask: (1, 1, Q, KV)
            attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, nq, d)
        return self.out_proj(out)


class CrossAttentionBlock(nn.Module):
    def __init__(self, qk_dim, num_heads, window_ratio=0.1):
        super().__init__()
        self.qk_dim = qk_dim
        self.num_heads = num_heads
        self.window_ratio = window_ratio
        self.conv_q = nn.Conv2d(
            qk_dim, qk_dim, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.cross_attn = FlexCrossAttention(qk_dim, num_heads)

    def forward(self, q, k, v):
        b, h_out, w_out, c_qk = q.shape
        _, h_kv, w_kv, c_v = v.shape
        ratio = self.window_ratio

        q_conv = self.conv_q(q.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        q_flat = q_conv.reshape(b, -1, c_qk)
        k_flat = k.reshape(b, -1, c_qk)
        v_flat = v.reshape(b, -1, c_v)

        approx_kv = int((2 * ratio * h_kv + 2) * w_kv)
        target_bytes = 256 * 1024 * 1024
        max_q = max(w_out, target_bytes // max(1, approx_kv * self.num_heads * 4))
        tile_rows = max(1, max_q // w_out)

        outputs = []
        for rs in range(0, h_out, tile_rows):
            re = min(rs + tile_rows, h_out)
            qt = q_flat[:, rs * w_out : re * w_out, :]

            r_lo = max(0.0, (rs + 0.5) / h_out - ratio)
            r_hi = min(1.0, (re - 0.5) / h_out + ratio)
            krs = max(0, int(math.floor(r_lo * h_kv)))
            kre = min(h_kv, int(math.ceil(r_hi * h_kv)))
            kt = k_flat[:, krs * w_kv : kre * w_kv, :]
            vt = v_flat[:, krs * w_kv : kre * w_kv, :]

            mask = tile_mask(
                rs, re, w_out, h_out, krs, kre, w_kv, h_kv, ratio, q.device
            )
            out = self.cross_attn(qt, kt, vt, mask=mask)
            outputs.append(out)

        out_seq = torch.cat(outputs, dim=1)
        return out_seq.reshape(b, h_out, w_out, c_v)


# ---------------------------- AnyUp main model ----------------------------
class AnyUp(nn.Module):
    def __init__(
        self,
        input_dim=3,
        qk_dim=128,
        kernel_size=1,
        kernel_size_lfu=5,
        window_ratio=0.1,
        num_heads=4,
    ):
        super().__init__()
        self.qk_dim = qk_dim
        self.window_ratio = window_ratio

        self.image_encoder = self._make_encoder(input_dim, kernel_size)
        self._image_encoder_pad = kernel_size // 2
        self.key_encoder = self._make_encoder(qk_dim, 1)
        self.query_encoder = self._make_encoder(qk_dim, 1)
        self.key_features_encoder = self._make_encoder(
            None,
            1,
            first_layer_k=kernel_size_lfu,
        )

        self.cross_decode = CrossAttentionBlock(
            qk_dim=qk_dim,
            num_heads=num_heads,
            window_ratio=window_ratio,
        )
        self.aggregation = self._make_encoder(2 * qk_dim, 3)
        self._aggregation_pad = 3 // 2
        self.pos_embed = AnyUpPositionalEmbedding(qk_dim)  # 修正点

        self.register_buffer(
            "imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 1, 3)
        )
        self.register_buffer(
            "imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 1, 3)
        )

    def _make_encoder(self, in_ch, k, layers=2, first_layer_k=0):
        parts = []
        if first_layer_k == 0:
            if in_ch is None:
                raise ValueError("in_ch must be specified for Conv2d")
            parts.append(nn.Conv2d(in_ch, self.qk_dim, k, padding=0, bias=False))
        else:
            parts.append(LearnedFeatureUnification(self.qk_dim, first_layer_k))
        for _ in range(layers):
            parts.append(
                ResBlock(self.qk_dim, self.qk_dim, kernel_size=1, num_groups=8)
            )
        return nn.ModuleList(parts)

    def _run_encoder(self, encoder, x, reflect_pad=0):
        # x: NHWC
        if reflect_pad > 0:
            x = reflect_pad(x, reflect_pad)
        x = x.permute(0, 3, 1, 2)  # to NCHW for Conv2d
        for layer in encoder:
            if isinstance(layer, LearnedFeatureUnification):
                x = x.permute(0, 2, 3, 1)  # to NHWC for LFU
                x = layer(x)
                x = x.permute(0, 3, 1, 2)  # back to NCHW
            else:
                x = layer(x)
        return x.permute(0, 2, 3, 1)  # back to NHWC

    def _normalize(self, x):
        return x / (torch.norm(x, dim=-1, keepdim=True) + 1e-8)

    def upsample(self, enc_img, feats, out_size):
        b, h, w, c = feats.shape
        oh, ow = out_size

        q = pool_to(self._run_encoder(self.query_encoder, enc_img), (oh, ow))
        k = pool_to(self._run_encoder(self.key_encoder, enc_img), (h, w))
        feats_norm = self._normalize(feats)
        k_feat = self._run_encoder(self.key_features_encoder, feats_norm)
        k = torch.cat([k, k_feat], dim=-1)
        k = self._run_encoder(self.aggregation, k, reflect_pad=self._aggregation_pad)
        v = feats
        return self.cross_decode(q, k, v)

    def forward(self, images, features, output_size=None):
        """
        images: NHWC, range [-1, 1] (tanh normalized)
        features: NHWC, arbitrary feature map (from any vision encoder)
        output_size: (H_out, W_out) tuple, defaults to image size
        Returns: NCHW tensor (compatible with downstream tasks)
        """
        if output_size is None:
            output_size = (images.shape[1], images.shape[2])

        # Preprocess images: [-1,1] -> [0,1] -> ImageNet norm
        images = images * 0.5 + 0.5
        images = (images - self.imagenet_mean) / self.imagenet_std
        images = images.to(features.dtype)

        # Encode image and add positional embedding
        enc = self._run_encoder(
            self.image_encoder, images, reflect_pad=self._image_encoder_pad
        )
        b, h_enc, w_enc, c_enc = enc.shape
        coords = create_coordinate(h_enc, w_enc, device=enc.device)  # (1, H*W, 2)
        enc_flat = enc.reshape(b, -1, c_enc)
        enc_flat = self.pos_embed(enc_flat, coords)  # 使用正确的 Sinusoidal Embedding
        enc = enc_flat.reshape(b, h_enc, w_enc, c_enc)

        result = self.upsample(enc, features, output_size)
        # Return NCHW for downstream usage
        return result.permute(0, 3, 1, 2)


# ---------------------------- Example Usage ----------------------------
if __name__ == "__main__":
    # 1. Init model
    model = AnyUp(
        input_dim=3,
        qk_dim=128,
        kernel_size=1,
        kernel_size_lfu=5,
        window_ratio=0.1,
        num_heads=4,
    )

    # 2. Load pretrained weights (optional, replace path with actual file if needed)
    # checkpoint = torch.load("path/to/anyup.pth")
    # model.load_state_dict(checkpoint["model_state_dict"])

    # 3. Dummy input
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    batch_size = 2
    img_size = (64, 64)  # low-res original image (spatial size of features)
    out_size = (224, 224)  # target high-res output size

    # Simulate output of any vision encoder (e.g., DINO, CLIP, MAE)
    # Shape: (B, H, W, C)
    low_res_features = torch.randn(
        batch_size, img_size[0], img_size[1], 768, device=device
    )

    # Original RGB image (range [-1, 1])
    original_image = (
        torch.randn(batch_size, img_size[0], img_size[1], 3, device=device) * 0.5
    )

    # 4. Run inference
    with torch.no_grad():
        upsampled_features = model(original_image, low_res_features, out_size)

    print(
        f"Input feature shape: {low_res_features.shape} -> Output shape: {upsampled_features.shape}"
    )
