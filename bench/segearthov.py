import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Fixed AdaptiveConv (no extra normalization)
# =========================
class FixedAdaptiveConv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, kernel):
        B, C, H_pad, W_pad = input.shape
        B, H_out, W_out, kH, kW = kernel.shape

        patches = F.unfold(input, kernel_size=(kH, kW), padding=0)
        patches = patches.view(B, C, kH * kW, H_out, W_out)

        kernel_flat = kernel.view(B, 1, H_out, W_out, kH * kW).permute(0, 1, 4, 2, 3)
        # No extra normalization here (the kernel is already softmax-normalized)
        output = (patches * kernel_flat).sum(dim=2)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError


# =========================
# Fixed JBU Core (removed inference-only constants)
# =========================
class FixedJBULearnedRange(nn.Module):
    def __init__(self, guidance_dim, feat_dim, key_dim, scale=2, radius=5):
        super().__init__()
        self.radius = radius
        self.diameter = self.radius * 2 + 1
        self.key_dim = key_dim

        self.range_temp = nn.Parameter(torch.tensor(-2.0))

        self.range_proj = nn.Sequential(
            nn.Conv2d(guidance_dim, key_dim, 1),
            nn.GELU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(key_dim, key_dim, 1),
        )

        self.fixup_proj = nn.Sequential(
            nn.Conv2d(guidance_dim + self.diameter**2, self.diameter**2, 1),
            nn.GELU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(self.diameter**2, self.diameter**2, 1),
        )

        self.sigma_spatial = nn.Parameter(torch.tensor(1.0))

    def get_range_kernel(self, x):
        B, C, H, W = x.shape

        proj = self.range_proj(x)
        proj_pad = F.pad(proj, [self.radius] * 4, mode="reflect")

        queries = (
            torch.nn.Unfold(self.diameter)(proj_pad)
            .view(B, -1, self.diameter * self.diameter, H, W)
            .permute(0, 1, 3, 4, 2)
        )

        temp = self.range_temp.exp().clamp(1e-4, 10.0)

        attn = torch.einsum("bchwp,bchw->bphw", queries, proj)
        attn = attn * temp
        # Removed: attn = attn + 0.1
        return F.softmax(attn, dim=1)

    def get_spatial_kernel(self, device):
        d = torch.linspace(-1, 1, self.diameter, device=device)
        x, y = torch.meshgrid(d, d, indexing="ij")
        grid = torch.stack([x, y], dim=0)
        return torch.exp(-(grid**2).sum(0) / (2 * self.sigma_spatial**2)).reshape(
            1, self.diameter * self.diameter, 1, 1
        )

    def forward(self, source, guidance):
        B, _, H, W = guidance.shape

        spatial = self.get_spatial_kernel(source.device)
        range_k = self.get_range_kernel(guidance)

        # soft spatial prior
        kernel = range_k * (0.5 + 0.5 * spatial)
        kernel = kernel / (kernel.sum(1, keepdim=True) + 1e-6)

        # Removed: kernel = kernel + 0.02 * self.fixup_proj(torch.cat([kernel, guidance], dim=1))

        kernel = kernel.permute(0, 2, 3, 1).reshape(
            B, H, W, self.diameter, self.diameter
        )

        hr = F.interpolate(source, (H, W), mode="bicubic", align_corners=False)
        hr_pad = F.pad(hr, [self.radius] * 4, mode="reflect")

        return FixedAdaptiveConv.apply(hr_pad, kernel)


# =========================
# Fixed JBU Wrapper (single upsampling, residual coefficient = 1.0)
# =========================
class FixedJBUOne(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.up = FixedJBULearnedRange(3, feat_dim, 32, radius=5)
        self.fixup_proj = nn.Sequential(
            nn.Dropout2d(0.2), nn.Conv2d(feat_dim, feat_dim, 1)
        )

    def upsample(self, source, guidance, up):
        # source: low-res feature, guidance: RGB (full-res)
        _, _, h, w = source.shape
        small = F.adaptive_avg_pool2d(guidance, (h * 2, w * 2))
        return up(source, small)

    def forward(self, source, guidance):
        # Single JBU upsampling instead of 4 iterations
        x = self.upsample(source, guidance, self.up)
        x = self.fixup_proj(x)

        src_hr = F.interpolate(
            source, size=x.shape[-2:], mode="bilinear", align_corners=False
        )
        # Residual coefficient changed from 0.1 to 1.0
        return src_hr + x


# =========================
# FeatUp Wrapper (compatible with DenseCLIP call style)
# =========================
class FixedFeatUpWrapper(nn.Module):
    def __init__(self, upsampler, input_dim, feat_dim, output_dim=768):
        super().__init__()
        self.adapter = nn.Conv2d(input_dim, feat_dim, 1)
        self.upsampler = upsampler
        self.proj_back = (
            nn.Identity()
            if feat_dim == output_dim
            else nn.Conv2d(feat_dim, output_dim, 1)
        )

    def forward(self, source, guidance):
        x = self.adapter(source)
        x = self.upsampler(x, guidance)
        return self.proj_back(x)

    def __call__(self, source, guidance):
        return self.forward(source, guidance)


# =========================
# Inspection helper (unchanged)
# =========================
def inspect_ckpt(state_dict):
    print("\n===== CHECKPOINT STRUCTURE =====\n")
    for k, v in state_dict.items():
        if "fixup_proj" in k or "range_proj" in k:
            print(f"{k:60s} -> {tuple(v.shape)}")

    print("\n===== KERNEL INFO =====\n")
    for k, v in state_dict.items():
        if "fixup_proj.0.weight" in k:
            out_c, in_c, kh, kw = v.shape
            print("kernel size:", kh, "x", kw)
            print("implied radius:", (kh - 1) // 2)


# =========================
# Loader (uses fixed classes)
# =========================
def load_featup_upsampler(
    ckpt_path="./simfeatup.ckpt", device="cuda", input_dim=768, output_dim=768
):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    cleaned = {}
    for k, v in state.items():
        cleaned[k[10:] if k.startswith("upsampler.") else k] = v

    # Infer feat_dim from checkpoint
    feat_dim = None
    for k in cleaned:
        if "fixup_proj.1.weight" in k:
            feat_dim = cleaned[k].shape[0]
            break
    if feat_dim is None:
        raise RuntimeError("Cannot infer feat_dim from checkpoint")

    inspect_ckpt(cleaned)
    print("Loaded feat_dim:", feat_dim)

    model = FixedJBUOne(feat_dim)
    model.load_state_dict(cleaned, strict=True)

    wrapper = FixedFeatUpWrapper(model, input_dim, feat_dim, output_dim)
    return wrapper.to(device).eval()


# =========================
# Example usage
# =========================
if __name__ == "__main__":
    # Simulate loading
    # upsampler = load_fixed_featup_upsampler("path/to/checkpoint.ckpt", device="cuda")
    # low_feat = torch.randn(1, 768, 16, 16).cuda()
    # rgb = torch.randn(1, 3, 224, 224).cuda()
    # high_feat = upsampler(low_feat, rgb)
    # print(high_feat.shape)
    pass
