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


def _compute_h_lim_w_lim(height, width):
    h_lim, w_lim = 1 - 1 / height, 1 - 1 / width
    return h_lim, w_lim


def create_coordinate_grid_2d(
    height: int = 16,
    width: int = 16,
    device=torch.device("cpu"),
):
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
    """
    x: B S C
    """
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
        patch_coords_lr: torch.Tensor,
        patch_coords_hr: torch.Tensor,
        pixels_lr: torch.Tensor,
        pixels_hr: torch.Tensor,
        init_sigma: float = 8.0,
        init_range: float = 0.12,
        init_orientation: float = 0.0,
        max_radius: int = 6,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.register_buffer("patch_coords_lr", patch_coords_lr)
        self.register_buffer("pixel_coords_hr", patch_coords_hr)
        self.register_buffer("pixels_lr", pixels_lr)
        self.register_buffer("pixels_hr", pixels_hr)

        self.max_radius = max_radius
        self.eps = eps

        self.batch_size, self.num_splats, self.ND = patch_coords_lr.shape

        # TODO generalize to ND
        if self.ND != 2:
            raise NotImplementedError()

        # B, H, L, ND
        spatial_differences = patch_coords_hr.unsqueeze(
            2
        ) - self.patch_coords_lr.unsqueeze(1)

        # euclidean distance
        # B H L ND -> B H L
        spatial_distances = spatial_differences.pow(2).mean(-1)

        # High resolution pixels only interact with the k nearest splats
        k = self.max_radius**self.ND

        # B H L -> B H K
        _, neighbor_idx = (-spatial_distances).topk(k, dim=2, sorted=False)

        # B H [L] ND, B H K -> B H K ND
        neighbor_diffs = spatial_differences.take_along_dim(
            neighbor_idx[:, :, :, None], 2
        )

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

    def _get_covariance_inverse(self):
        """
        Computes Sigma^{-1} for the Mahalanobis distance.
        Sigma = R @ [[sx^2, 0], [0, sy^2]] @ R.T
        Sigma^{-1} = R @ [[1/sx^2, 0], [0, 1/sy^2]] @ R.T
        """
        sigmas = torch.exp(self.log_sigma)  # (B, N, 2)
        sx, sy = sigmas[..., 0], sigmas[..., 1]

        # Rotation Matrix R
        # [[cos, -sin], [sin, cos]]
        cos_theta = torch.cos(self.theta).squeeze(-1)  # (B, N)
        sin_theta = torch.sin(self.theta).squeeze(-1)

        # Construct Inverse Scaling Matrix diagonals (1/s^2)
        inv_sx2 = 1.0 / (sx**2 + self.eps)
        inv_sy2 = 1.0 / (sy**2 + self.eps)

        # Manual matrix multiplication for R * S_inv * R.T to keep it fast/explicit
        # Let A = Sigma^{-1}
        # A_00 = cos^2/sx^2 + sin^2/sy^2
        a00 = (cos_theta**2 * inv_sx2) + (sin_theta**2 * inv_sy2)
        # A_01 = cos*(-sin)/sx^2 + (-sin)*cos/sy^2  => sin*cos*(1/sy^2 - 1/sx^2)
        a01 = cos_theta * sin_theta * (inv_sy2 - inv_sx2)
        # A_11 = sin^2/sx^2 + cos^2/sy^2
        a11 = (sin_theta**2 * inv_sx2) + (cos_theta**2 * inv_sy2)

        # Stack into (B, N, 2, 2)
        row1 = torch.stack([a00, a01], dim=-1)
        row2 = torch.stack([a01, a11], dim=-1)  # Symmetric
        sigma_inv = torch.stack([row1, row2], dim=-2)

        return sigma_inv

    def compute_sparse_weights(self):
        """
        Calculates the mixing weights w_{p <- q}
        target_coords: (B H, ND) - Coordinates of output pixels
        """

        neighbor_idx = self.neighbor_idx
        neighbor_diffs = self.neighbor_diffs

        # Get Sigma Inverse: (B, L, ND, ND)
        sigma_inv = self._get_covariance_inverse()

        # Mahalanobis Distance: -0.5 * delta.T * Sigma^-1 * delta
        # B [L] ND ND, B H K -> B H K ND ND
        sigma_inv = sigma_inv.unsqueeze(1).take_along_dim(
            neighbor_idx[:, :, :, None, None], 2
        )

        # Inner term: delta * Sigma_inv
        term = einsum(neighbor_diffs, sigma_inv, "b h k d, b h k d dd -> b h k dd")

        mahalanobis = term * neighbor_diffs
        # Quadratic form: 0.5 * sum(term * diff_spatial)
        mahalanobis = 0.5 * reduce(mahalanobis, "b h k nd -> b h k", "sum")

        log_w_s = -mahalanobis

        # Range Weight Calculation (log space)
        # || I(p) - I(q) ||^2 / (2 * sigma_r^2)

        # B [L] C, B H K -> B H K C
        selected_pixels_lr = self.pixels_lr[:, None, :, :].take_along_dim(
            neighbor_idx[:, :, :, None], 2
        )

        # B H C, B H K C -> B H K C
        diff_color = self.pixels_hr.unsqueeze(2) - selected_pixels_lr

        # B H K C -> B H K
        dist_sq_color = (diff_color**2).sum(dim=-1)

        sigma_r = torch.exp(self.log_sigma_r)

        # B [L], B H K -> B H K
        selected_sigma_r = sigma_r[:, None, :].take_along_dim(neighbor_idx, 2)

        # B H K, B H K -> B H K
        log_w_r = -dist_sq_color / (2 * selected_sigma_r**2 + self.eps)

        log_w_total = log_w_s + log_w_r

        # Normalize (Softmax over neighbors q)
        # B H K
        w_p_q = F.softmax(log_w_total, dim=-1)

        return w_p_q

    def forward(self, values_lr: torch.Tensor):
        # Calculate sparse mixing weights based on spatial + guidance similarity
        # (B H K, B H K)
        weights = self.compute_sparse_weights()

        neighbor_idx = self.neighbor_idx

        *_, D = values_lr.shape
        B, H, K = neighbor_idx.shape

        # If H is small, this method is effective.
        # However, if H is large (~100k) This uses to much memory...
        # B [L] D, B H K -> B H K D
        # selected_values_lr = values_lr[:, None, :, :].take_along_dim(
        #     idx.unsqueeze(-1), 2
        # )
        # # Apply weights: F_hr(p) = sum_q ( w_{p<-q} * F_lr(q) )
        # output = einsum(weights, selected_values_lr, "b h k, b h k d -> b h d")

        # Simply loop over k which is small (16)
        output = torch.zeros((B, H, D), dtype=values_lr.dtype, device=values_lr.device)
        for k in range(K):
            idx_k = neighbor_idx[:, :, k]
            # weight_k shape: (B, H, 1) for broadcasting
            weight_k = weights[:, :, k].unsqueeze(-1)

            # 3. Gather only the current neighbor's values
            # We need to gather from values_lr (B, L, D) using idx_k (B, H)
            # Resulting gathered_values shape: (B, H, D)

            # Expand indices to match D dimension for gather
            # (B, H, D)
            gather_idx = idx_k.unsqueeze(-1).expand(-1, -1, D)

            gathered_values = torch.gather(values_lr, 1, gather_idx)

            # 4. Accumulate weighted sum directly into output
            # This avoids creating the massive (B, H, K, D) tensor
            output = output + gathered_values * weight_k

        return output

    def fit(self, steps: int = 50, lr: float = 1e2, callback=None):
        optimizer = optim.SGD(self.parameters(), lr=lr)

        for i in range(steps):
            optimizer.zero_grad()

            # Reconstruct HR image using LR image as both Value and Guidance Source
            # Note: In TTO, "values_lr" is the Low Res RGB image.
            pred_hr = self.forward(values_lr=self.pixels_lr)

            loss = F.l1_loss(pred_hr, self.pixels_hr)

            loss.backward()
            optimizer.step()

            if callback is not None:
                callback(i, loss.detach())


def main(
    image_path: Path = Path("asset/9.png"),
    output_path: Path = Path("output.png"),
    resize_size: tuple[int, int] = (224, 224),
    device_str="cpu",
    num_optimization_steps: int = 20,
    lr: float = 1e2,
):
    device = torch.device(device_str)
    torch.manual_seed(42)

    image_encoder = transformers.AutoModel.from_pretrained("facebook/dinov2-base").to(
        device
    )

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

    image_lr = F.interpolate(
        image_hr,
        (nph, npw),
        mode="bilinear",
        align_corners=False,
    )
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

    # Compute PCA using low res features
    # you can also compute PCA using high res features but it would take longer
    *_, pca_v_proj = torch.pca_lowrank(features_lr, q=3, niter=20)

    with torch.inference_mode():
        # B (NPH NPW) D -> B (H W) D
        features_upsampled = feature_upsampler.forward(features_lr)

    features_upsampled = einsum(
        features_upsampled, pca_v_proj, "b h_w d, b d c -> b h_w c"
    )

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

        # 1. 构造坐标和数据（与之前一致）
        patch_coords_hr = create_coordinate_grid_2d(H, W, device)
        patch_coords_hr = repeat(patch_coords_hr, "h w nd -> b (h w) nd", b=B)

        patch_coords_lr = create_coordinate_grid_2d(h, w, device)
        patch_coords_lr = repeat(patch_coords_lr, "nph npw nd -> b (nph npw) nd", b=B)

        pixels_hr = rearrange(guide, "b c h w -> b (h w) c")
        guide_lr = F.interpolate(
            guide, size=(h, w), mode="bilinear", align_corners=False
        )
        pixels_lr = rearrange(guide_lr, "b c h w -> b (h w) c")

        # 2. 强行开启梯度上下文，确保 fit 内部的 loss.backward() 正常工作
        # 哪怕外层套了 @torch.no_grad()，这里也能独立计算梯度
        with torch.enable_grad():
            upsampler = GaussianFeatureUpsampler(
                patch_coords_lr=patch_coords_lr,
                patch_coords_hr=patch_coords_hr,
                pixels_lr=pixels_lr,
                pixels_hr=pixels_hr,
            ).to(device)

            # 进行内部的测试时迭代优化
            upsampler.fit(steps=self.steps, lr=self.lr, callback=None)

        # 3. 优化完成后，提取上采样特征（推理阶段，不需要外层梯度）
        features_lr_flat = rearrange(lr_features, "b c h w -> b (h w) c")

        # 使用 detach() 彻底切断 CLIP 主网络与上采样器之间的梯度干扰
        with torch.no_grad():
            features_upsampled = upsampler.forward(features_lr_flat.detach())

        out = rearrange(features_upsampled, "b (h w) c -> b c h w", h=H, w=W)
        return out


if __name__ == "__main__":
    jsonargparse.CLI(main)
