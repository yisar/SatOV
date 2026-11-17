from pathlib import Path
import math
import transformers

import torch
from tqdm import tqdm
from torch import nn, optim
import torch.nn.functional as F
import torchvision
from einops import rearrange, reduce, repeat, einsum
import jsonargparse

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def _compute_h_lim_w_lim(height, width):
    h_lim, w_lim = 1 - 1 / height, 1 - 1 / width
    return h_lim, w_lim


def create_coordinate_grid_2d(height=16, width=16, device=torch.device("cpu")):
    dtype = torch.float32
    h_lim, w_lim = _compute_h_lim_w_lim(height, width)
    coordinates = torch.meshgrid(
        [
            torch.linspace(-h_lim, h_lim, height, device=device, dtype=dtype),
            torch.linspace(-w_lim, w_lim, width, device=device, dtype=dtype),
        ],
        indexing="ij",
    )
    coordinates = torch.stack(coordinates, -1)
    return coordinates


def min_max_scale(x: torch.Tensor, eps=1e-6):
    min = x.amin(1, keepdim=True)
    max = x.amax(1, keepdim=True)
    x = (x - min) / (max - min).clip(eps)
    x = x.clip(0, 1)
    return x


class GaussianFeatureUpsampler(nn.Module):
    patch_coords_lr: torch.Tensor
    patch_coords_hr: torch.Tensor
    pixels_lr: torch.Tensor
    pixels_hr: torch.Tensor
    neighbor_idx: torch.Tensor
    neighbor_diffs: torch.Tensor

    def __init__(
        self,
        patch_coords_lr,
        patch_coords_hr,
        pixels_lr,
        pixels_hr,
        init_sigma=0.12,
        init_range=0.12,
        init_orientation=0.0,
        max_radius=6,
        eps=1e-6,
    ):
        super().__init__()

        self.register_buffer("patch_coords_lr", patch_coords_lr)
        self.register_buffer("pixel_coords_hr", patch_coords_hr)
        self.register_buffer("pixels_lr", pixels_lr)
        self.register_buffer("pixels_hr", pixels_hr)

        self.max_radius = max_radius
        self.eps = eps
        self.batch_size, self.num_splats, self.ND = patch_coords_lr.shape
        if self.ND != 2:
            raise NotImplementedError()

        spatial_differences = patch_coords_hr.unsqueeze(2) - patch_coords_lr.unsqueeze(1)
        spatial_distances = spatial_differences.pow(2).mean(-1)
        k = self.max_radius ** self.ND
        _, neighbor_idx = (-spatial_distances).topk(k, dim=2, sorted=False)
        neighbor_diffs = spatial_differences.take_along_dim(neighbor_idx[:, :, :, None], 2)

        self.register_buffer("neighbor_idx", neighbor_idx)
        self.register_buffer("neighbor_diffs", neighbor_diffs)

        self.log_sigma = nn.Parameter(
            torch.ones(self.batch_size, self.num_splats, self.ND) * math.log(init_sigma)
        )
        self.theta = nn.Parameter(
            torch.ones(self.batch_size, self.num_splats, 1) * init_orientation
        )
        self.log_sigma_r = nn.Parameter(
            torch.ones(self.batch_size, self.num_splats) * math.log(init_range)
        )

        # ============================================================
        # Warp-2DGS: monotonic PWL remapping parameters
        # 每个 surfel 每轴一个 knot (c1, d1)，锚定 (-1,-1)->(1,1) 恒等初始化。
        # 初始时 c1=0, d1=0 -> D(t)=t -> 恒等映射，起点行为与原版一致。
        # c1 用 sigmoid 约束在 (-1,1) 内，d1 被裁剪到 [-1,1]。
        # ============================================================
        self.warp_c1 = nn.Parameter(torch.zeros(self.batch_size, self.num_splats, self.ND))
        self.warp_d1 = nn.Parameter(torch.zeros(self.batch_size, self.num_splats, self.ND))

    def _get_covariance_inverse(self):
        sigmas = torch.exp(self.log_sigma)
        sx, sy = sigmas[..., 0], sigmas[..., 1]
        cos_theta = torch.cos(self.theta).squeeze(-1)
        sin_theta = torch.sin(self.theta).squeeze(-1)
        inv_sx2 = 1.0 / (sx ** 2 + self.eps)
        inv_sy2 = 1.0 / (sy ** 2 + self.eps)
        a00 = (cos_theta ** 2 * inv_sx2) + (sin_theta ** 2 * inv_sy2)
        a01 = cos_theta * sin_theta * (inv_sy2 - inv_sx2)
        a11 = (sin_theta ** 2 * inv_sx2) + (cos_theta ** 2 * inv_sy2)
        row1 = torch.stack([a00, a01], dim=-1)
        row2 = torch.stack([a01, a11], dim=-1)
        return torch.stack([row1, row2], dim=-2)

    # ============================================================
    # NEW: Warp-2DGS monotonic piecewise-linear remapping
    # D(t): [-1,c1] -> [-1,d1], [c1,1] -> [d1,1], outside [-1,1] identity
    # ============================================================
    def _monotonic_pwl(self, t, c1, d1):
        """
        t:  (B, N, K) or broadcastable
        c1: (B, N, 1)  knot position, in (-1, 1)
        d1: (B, N, 1)  knot value,    in (-1, 1)
        returns warped t, same shape
        """
        c1 = torch.sigmoid(c1) * 2 - 1  # (-1, 1)
        d1 = torch.tanh(d1)  # (-1, 1)

        # slope of left segment: (-1 -> c1) maps to (-1 -> d1)
        slope_left = (d1 + 1) / (c1 + 1 + self.eps)
        # slope of right segment: (c1 -> 1) maps to (d1 -> 1)
        slope_right = (1 - d1) / (1 - c1 + self.eps)

        left = -1 + slope_left * (t + 1)
        right = d1 + slope_right * (t - c1)

        out = torch.where(t <= c1, left, right)
        # outside [-1, 1] identity (Gaussian tail untouched)
        out = torch.where(t.abs() > 1, t, out)
        return out

    def _warped_mahalanobis(self, mahalanobis):
        """
        Apply per-axis monotonic PWL remapping to the *signed* 1D coordinate
        before squaring. We recover a signed per-axis coordinate by projecting
        neighbor_diffs onto the principal axes of Sigma^{-1}, but since the
        Mahalanobis quadratic form already mixes axes, the cleanest Warp-2DGS
        analogue is to warp the scalar sqrt(mahalanobis) with per-axis knots
        aggregated (mean over axes) — this preserves the monotonic, identity-
        initialized behavior while still letting the kernel become asymmetric.
        """
        # mahalanobis: (B, H, K)
        t = torch.sqrt(mahalanobis.clamp(min=0) + self.eps)  # (B, H, K)

        # aggregate per-axis knots to a per-surfel scalar knot (mean over ND)
        c1 = self.warp_c1.mean(-1, keepdim=True)  # (B, N, 1)
        d1 = self.warp_d1.mean(-1, keepdim=True)

        # select knots for the neighbors of each HR pixel
        selected_c1 = c1[:, None, :, :].take_along_dim(
            self.neighbor_idx[:, :, :, None], 2
        ).squeeze(-1)  # (B, H, K)
        selected_d1 = d1[:, None, :, :].take_along_dim(
            self.neighbor_idx[:, :, :, None], 2
        ).squeeze(-1)

        t_warped = self._monotonic_pwl(t, selected_c1, selected_d1)
        return 0.5 * t_warped ** 2

    def compute_sparse_weights(self):
        neighbor_idx = self.neighbor_idx
        neighbor_diffs = self.neighbor_diffs

        sigma_inv = self._get_covariance_inverse()
        sigma_inv = sigma_inv.unsqueeze(1).take_along_dim(
            neighbor_idx[:, :, :, None, None], 2
        )

        term = einsum(neighbor_diffs, sigma_inv, "b h k d, b h k d dd -> b h k dd")
        mahalanobis = term * neighbor_diffs
        mahalanobis = 0.5 * reduce(mahalanobis, "b h k nd -> b h k", "sum")

        # ---- Warp-2DGS: replace plain Mahalanobis with warped version ----
        mahalanobis_warped = self._warped_mahalanobis(mahalanobis)
        log_w_s = -mahalanobis_warped
        # ------------------------------------------------------------------

        selected_pixels_lr = self.pixels_lr[:, None, :, :].take_along_dim(
            neighbor_idx[:, :, :, None], 2
        )
        diff_color = self.pixels_hr.unsqueeze(2) - selected_pixels_lr
        dist_sq_color = (diff_color ** 2).sum(dim=-1)

        sigma_r = torch.exp(self.log_sigma_r)
        selected_sigma_r = sigma_r[:, None, :].take_along_dim(neighbor_idx, 2)
        log_w_r = -dist_sq_color / (2 * selected_sigma_r ** 2 + self.eps)

        log_w_total = log_w_s + log_w_r
        w_p_q = F.softmax(log_w_total, dim=-1)
        return w_p_q

    def forward(self, values_lr: torch.Tensor):
        weights = self.compute_sparse_weights()
        neighbor_idx = self.neighbor_idx
        *_, D = values_lr.shape
        B, H, K = neighbor_idx.shape

        output = torch.zeros((B, H, D), dtype=values_lr.dtype, device=values_lr.device)
        for k in range(K):
            idx_k = neighbor_idx[:, :, k]
            weight_k = weights[:, :, k].unsqueeze(-1)
            gather_idx = idx_k.unsqueeze(-1).expand(-1, -1, D)
            gathered_values = torch.gather(values_lr, 1, gather_idx)
            output = output + gathered_values * weight_k
        return output

    def fit(self, steps: int = 50, lr: float = 1e2, callback=None):
        optimizer = optim.SGD(self.parameters(), lr=lr)
        for i in range(steps):
            optimizer.zero_grad()
            pred_hr = self.forward(values_lr=self.pixels_lr)
            loss = F.l1_loss(pred_hr, self.pixels_hr)
            loss.backward()
            optimizer.step()
            if callback is not None:
                callback(i, loss.detach())


def main(
    image_path: Path = Path("asset/4072.png"),
    output_path: Path = Path("output.png"),
    resize_size: tuple[int, int] = (224, 224),
    device_str="cpu",
    num_optimization_steps: int = 20,
    lr: float = 1e2,
):
    device = torch.device(device_str)
    torch.manual_seed(42)

    image_encoder = transformers.AutoModel.from_pretrained("facebook/dinov2-base").to(device)
    patch_size = image_encoder.config.patch_size
    image_encoder_processor = transformers.AutoImageProcessor.from_pretrained(
        "facebook/dinov2-base"
    )
    image_mean = torch.tensor(image_encoder_processor.image_mean).to(device)
    image_std = torch.tensor(image_encoder_processor.image_std).to(device)

    image_hr = torchvision.io.read_image(str(image_path)).to(device)
    image_hr = image_hr[:3].unsqueeze(0).div(255)
    image_hr = torchvision.transforms.Resize(resize_size)(image_hr)

    h, w = resize_size
    nph, npw = h // patch_size, w // patch_size

    patch_coords_hr = create_coordinate_grid_2d(h, w, device)
    patch_coords_hr = repeat(patch_coords_hr, "h w nd -> b (h w) nd", b=1)
    patch_coords_lr = create_coordinate_grid_2d(nph, npw, device)
    patch_coords_lr = repeat(patch_coords_lr, "nph npw nd -> b (nph npw) nd", b=1)

    image_lr = F.interpolate(image_hr, (nph, npw), mode="bilinear", align_corners=False)
    pixels_lr = rearrange(image_lr, "b c nph npw -> b (nph npw) c")
    pixels_hr = rearrange(image_hr, "b c h w -> b (h w) c")

    feature_upsampler = GaussianFeatureUpsampler(
        patch_coords_lr=patch_coords_lr,
        patch_coords_hr=patch_coords_hr,
        pixels_lr=pixels_lr,
        pixels_hr=pixels_hr,
    ).to(device)

    prog_bar = tqdm(total=num_optimization_steps)
    def _callback(step, loss):
        tqdm.write(f"{step:04} {loss:.5f}")
        prog_bar.update()

    feature_upsampler.fit(steps=num_optimization_steps, lr=lr, callback=_callback)
    prog_bar.close()

    image_hr_normalized = (image_hr - image_mean[None, :, None, None]) / image_std[
        None, :, None, None
    ]
    with torch.inference_mode():
        features_lr = image_encoder(pixel_values=image_hr_normalized).last_hidden_state
    features_lr = features_lr[:, -nph * npw :]

    *_, pca_v_proj = torch.pca_lowrank(features_lr, q=3, niter=20)
    with torch.inference_mode():
        features_upsampled = feature_upsampler.forward(features_lr)

    features_upsampled = einsum(features_upsampled, pca_v_proj, "b h_w d, b d c -> b h_w c")
    features_upsampled = (
        min_max_scale(features_upsampled).mul(255).round().to(torch.uint8).cpu()
    )
    features_upsampled = rearrange(features_upsampled, "b (h w) c -> b c h w", h=h, w=w)
    features_upsampled = torch.cat(
        (image_hr.mul(255).round().to(torch.uint8).cpu(), features_upsampled), 2
    )
    torchvision.io.write_png(features_upsampled[0], str(output_path))


class GaussianUpsamplerWrapper(nn.Module):
    def __init__(self, steps: int = 20, lr: float = 1e2):
        super().__init__()
        self.steps = steps
        self.lr = lr

    def forward(self, guide: torch.Tensor, lr_features: torch.Tensor):
        B, C, h, w = lr_features.shape
        _, _, H, W = guide.shape
        device = guide.device

        patch_coords_hr = create_coordinate_grid_2d(H, W, device)
        patch_coords_hr = repeat(patch_coords_hr, "h w nd -> b (h w) nd", b=B)
        patch_coords_lr = create_coordinate_grid_2d(h, w, device)
        patch_coords_lr = repeat(patch_coords_lr, "nph npw nd -> b (nph npw) nd", b=B)

        pixels_hr = rearrange(guide, "b c h w -> b (h w) c")
        guide_lr = F.interpolate(guide, size=(h, w), mode="bilinear", align_corners=False)
        pixels_lr = rearrange(guide_lr, "b c h w -> b (h w) c")

        with torch.enable_grad():
            upsampler = GaussianFeatureUpsampler(
                patch_coords_lr=patch_coords_lr,
                patch_coords_hr=patch_coords_hr,
                pixels_lr=pixels_lr,
                pixels_hr=pixels_hr,
            ).to(device)
            upsampler.fit(steps=self.steps, lr=self.lr, callback=None)

        features_lr_flat = rearrange(lr_features, "b c h w -> b (h w) c")
        with torch.no_grad():
            features_upsampled = upsampler.forward(features_lr_flat.detach())

        out = rearrange(features_upsampled, "b (h w) c -> b c h w", h=H, w=W)
        return out


if __name__ == "__main__":
    jsonargparse.CLI(main)