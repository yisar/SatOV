import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# AdaptiveConv (stable inference version)
# =========================
class AdaptiveConv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, kernel):
        B, C, H_pad, W_pad = input.shape
        B, H_out, W_out, kH, kW = kernel.shape

        patches = F.unfold(input, kernel_size=(kH, kW), padding=0)
        patches = patches.view(B, C, kH * kW, H_out, W_out)

        kernel_flat = kernel.view(B, 1, H_out, W_out, kH * kW).permute(0, 1, 4, 2, 3)

        # stability only
        kernel_flat = kernel_flat / (kernel_flat.sum(dim=2, keepdim=True) + 1e-6)

        output = (patches * kernel_flat).sum(dim=2)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError


# =========================
# JBU CORE (STRICT COMPATIBLE)
# =========================
class JBULearnedRange(nn.Module):
    def __init__(self, guidance_dim, feat_dim, key_dim, scale=2, radius=5):
        super().__init__()

        # MUST match checkpoint
        self.radius = radius
        self.diameter = self.radius * 2 + 1

        self.key_dim = key_dim

        # stable temperature (do NOT change structure)
        self.range_temp = nn.Parameter(torch.tensor(-2.0))

        self.range_proj = nn.Sequential(
            nn.Conv2d(guidance_dim, key_dim, 1),
            nn.GELU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(key_dim, key_dim, 1),
        )

        # MUST match checkpoint exactly
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

        # stability fix (IMPORTANT)
        attn = attn + 0.1

        return F.softmax(attn, dim=1)

    def get_spatial_kernel(self, device):
        d = torch.linspace(-1, 1, self.diameter, device=device)
        x, y = torch.meshgrid(d, d, indexing="ij")

        grid = torch.stack([x, y], dim=0)

        return torch.exp(
            -(grid ** 2).sum(0) / (2 * self.sigma_spatial**2)
        ).reshape(1, self.diameter * self.diameter, 1, 1)

    def forward(self, source, guidance):
        B, C, H, W = guidance.shape

        spatial = self.get_spatial_kernel(source.device)
        range_k = self.get_range_kernel(guidance)

        # soft spatial prior (DO NOT hard multiply)
        kernel = range_k * (0.5 + 0.5 * spatial)

        kernel = kernel / (kernel.sum(1, keepdim=True) + 1e-6)

        # weak fixup
        kernel = kernel + 0.02 * self.fixup_proj(
            torch.cat([kernel, guidance], dim=1)
        )

        kernel = kernel.permute(0, 2, 3, 1).reshape(
            B, H, W, self.diameter, self.diameter
        )

        hr = F.interpolate(source, (H, W), mode="bicubic", align_corners=False)
        hr_pad = F.pad(hr, [self.radius] * 4, mode="reflect")

        return AdaptiveConv.apply(hr_pad, kernel)


# =========================
# JBU wrapper (CRITICAL FIX: nn.Module + callable compatibility)
# =========================
class JBUOne(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()

        self.up = JBULearnedRange(3, feat_dim, 32, radius=5)

        self.fixup_proj = nn.Sequential(
            nn.Dropout2d(0.2),
            nn.Conv2d(feat_dim, feat_dim, 1)
        )

    def upsample(self, source, guidance, up):
        _, _, h, w = source.shape
        small = F.adaptive_avg_pool2d(guidance, (h * 2, w * 2))
        return up(source, small)

    def forward(self, source, guidance):
        x = source
        src = source

        for _ in range(4):
            x = self.upsample(x, guidance, self.up)

        x = self.fixup_proj(x)

        src_hr = F.interpolate(src, size=x.shape[-2:], mode="bilinear", align_corners=False)

        return src_hr + 0.1 * x


# =========================
# FeatUp Wrapper (FIXED: callable compatible)
# =========================
class FeatUpWrapper(nn.Module):
    def __init__(self, upsampler, input_dim, feat_dim, output_dim=768):
        super().__init__()

        self.adapter = nn.Conv2d(input_dim, feat_dim, 1)
        self.upsampler = upsampler

        self.proj_back = (
            nn.Identity()
            if feat_dim == output_dim
            else nn.Conv2d(feat_dim, output_dim, 1)
        )

    # ⭐ CRITICAL FIX: make it compatible with lambda g,f call style
    def forward(self, source, guidance):
        x = self.adapter(source)
        x = self.upsampler(x, guidance)
        return self.proj_back(x)

    # ⭐ CRITICAL FIX: DenseCLIP calls like hub_model(f,g)
    def __call__(self, source, guidance):
        return self.forward(source, guidance)
    
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
# LOADER (STRICT SAFE, NO SHAPE CHANGE)
# =========================
def load_featup_upsampler(
    ckpt_path="./simfeatup.ckpt",
    device="cuda",
    input_dim=768,
    output_dim=768
):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    cleaned = {}
    for k, v in state.items():
        cleaned[k[10:] if k.startswith("upsampler.") else k] = v

    feat_dim = None
    for k in cleaned:
        if "fixup_proj.1.weight" in k:
            feat_dim = cleaned[k].shape[0]
            break

    inspect_ckpt(cleaned)


    print("Loaded feat_dim:", feat_dim)

    model = JBUOne(feat_dim)
    model.load_state_dict(cleaned, strict=True)

    wrapper = FeatUpWrapper(model, input_dim, feat_dim, output_dim)

    return wrapper.to(device).eval()