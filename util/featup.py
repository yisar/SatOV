import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# JBU CORE MODULE
# =========================
class JBULearnedRange(nn.Module):
    def __init__(self, guidance_dim, feat_dim, key_dim=32, radius=3):
        super().__init__()
        self.radius = radius
        self.d = radius * 2 + 1
        self.key_dim = key_dim

        self.range_proj = nn.Sequential(
            nn.Conv2d(guidance_dim, key_dim, 1),
            nn.GELU(),
            nn.Conv2d(key_dim, key_dim, 1),
        )

        self.range_temp = nn.Parameter(torch.tensor(1.0))
        self.sigma_spatial = nn.Parameter(torch.tensor(1.0))
        self.fixup_proj = nn.Conv2d(feat_dim, feat_dim, 1)

    def get_spatial_kernel(self, device):
        dist = torch.linspace(-1, 1, self.d, device=device)
        x, y = torch.meshgrid(dist, dist, indexing="ij")
        patch = x**2 + y**2

        sigma = self.sigma_spatial.abs() + 1e-4
        kernel = torch.exp(-patch / (2 * sigma**2))

        return kernel.view(1, 1, self.d * self.d, 1, 1)

    def get_range_kernel(self, guidance):
        B, C, H, W = guidance.shape

        q = self.range_proj(guidance)
        q_pad = F.pad(q, [self.radius] * 4, mode="reflect")

        patches = F.unfold(q_pad, kernel_size=self.d)
        patches = patches.view(B, self.key_dim, self.d * self.d, H, W)

        center = q.unsqueeze(2)
        sim = (patches * center).sum(dim=1)

        sim = sim / (self.range_temp.abs() + 1e-4)
        return torch.softmax(sim, dim=1).unsqueeze(1)

    def forward(self, source, guidance):
        B, C, H, W = source.shape

        Hh, Wh = H * 2, W * 2

        source_hr = F.interpolate(
            source, (Hh, Wh), mode="bilinear", align_corners=False
        )
        guidance_hr = F.interpolate(
            guidance, (Hh, Wh), mode="bilinear", align_corners=False
        )

        spatial = self.get_spatial_kernel(source.device)
        range_k = self.get_range_kernel(guidance_hr)

        kernel = spatial * range_k
        kernel = kernel / (kernel.sum(dim=2, keepdim=True) + 1e-6)
        kernel = kernel.squeeze(1)

        unfolded = F.unfold(source_hr, kernel_size=self.d, padding=self.radius)
        unfolded = unfolded.view(B, C, self.d * self.d, Hh, Wh)

        out = (unfolded * kernel.unsqueeze(1)).sum(dim=2)

        return self.fixup_proj(out)


# =========================
# UPSAMPLER (FINAL SAFE VERSION)
# =========================
class JBUOne(nn.Module):
    def __init__(self, in_channels=None, feat_dim=512):
        super().__init__()

        self.feat_dim = feat_dim

        # 🔥 ONLY SAFE OPTION: fully dynamic projection
        # in_channels=None => LazyConv2d handles everything
        self.adapter = nn.LazyConv2d(feat_dim, kernel_size=1)

        self.up = JBULearnedRange(3, feat_dim)

        self.refine = nn.Sequential(nn.Dropout2d(0.2), nn.Conv2d(feat_dim, feat_dim, 1))

    def upsample(self, x, guidance):
        _, _, h, w = x.shape

        guidance_hr = F.interpolate(
            guidance, size=(h * 2, w * 2), mode="bilinear", align_corners=False
        )

        return self.up(x, guidance_hr)

    def forward(self, source, guidance):

        # 🔥 SAFE projection (no assumptions on channels)
        src_feat = self.adapter(source)

        x = src_feat

        # 16x upsampling
        for _ in range(4):
            x = self.upsample(x, guidance)

        x = self.refine(x)

        src_feat = F.interpolate(
            src_feat, size=x.shape[-2:], mode="bilinear", align_corners=False
        )

        return x * 0.1 + src_feat


# =========================
# TEST
# =========================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    B = 2
    C = 768  # can be ANYTHING: 512 / 768 / 1024 / unknown
    h, w = 16, 16

    low_res_feat = torch.randn(B, C, h, w).to(device)
    guide_rgb = torch.randn(B, 3, h * 16, w * 16).to(device)

    model = JBUOne(in_channels=None, feat_dim=512).to(device).eval()

    with torch.no_grad():
        out = model(low_res_feat, guide_rgb)

    print("input:", low_res_feat.shape)
    print("output:", out.shape)

    assert out.shape[-2:] == (h * 16, w * 16)
    print("✅ SUCCESS: fully dynamic JBU pipeline (no channel dependency)")
