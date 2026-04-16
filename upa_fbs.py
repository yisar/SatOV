import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import time

# --- 1. 核心算子：可微分的 Fast Bilateral Solver (FBS) ---

class DifferentiableFBS(nn.Module):
    """
    基于迭代亲和力传播的可微 FBS 算子
    作为 UPA 后的后处理层，嵌入计算图中共同参与优化
    """
    def __init__(self, lam=64.0, sigma_luma=0.1, iters=3):
        super().__init__()
        self.lam = lam
        self.sigma_luma = sigma_luma
        self.iters = iters

    def forward(self, target, guide):
        B, C, H, W = target.shape
        refined = target
        
        # 为了提高效率，这里使用 3x3 的局部双边邻域
        for _ in range(self.iters):
            # 1. 提取引导图邻域进行颜色亲和力计算
            unfolded_guide = F.unfold(guide, kernel_size=3, padding=1) # [B, 3*9, H*W]
            center_guide = guide.view(B, 3, 1, -1)
            neighbor_guide = unfolded_guide.view(B, 3, 9, -1)
            
            # 计算相似度权重 (高斯核)
            diff = torch.exp(-torch.sum((neighbor_guide - center_guide)**2, dim=1) / (2 * self.sigma_luma**2)) # [B, 9, H*W]
            weight = diff / (diff.sum(dim=1, keepdim=True) + 1e-5)
            
            # 2. 提取目标图邻域并加和
            unfolded_target = F.unfold(refined, kernel_size=3, padding=1) # [B, C*9, H*W]
            neighbor_target = unfolded_target.view(B, C, 9, -1)
            
            # 执行加权聚合
            refined_step = torch.sum(neighbor_target * weight.unsqueeze(1), dim=2).view(B, C, H, W)
            
            # 3. 能量平衡项 (Energy Balance)
            # lam 越大，越依赖平滑后的结果；alpha 控制原始信息的保留比例
            alpha = 1.0 / (1.0 + self.lam * 0.05)
            refined = alpha * target + (1 - alpha) * refined_step
            
        return refined

# --- 2. 核心算子：可微分的双边网格 (UPA Core) ---

def _tanh_bound_pi(raw):
    return math.pi * torch.tanh(raw)

def gs_jbu_grid_differentiable(feat_lr, guide_hr, sx, sy, th, sr, num_bins=12):
    B, C, Hl, Wl = feat_lr.shape
    _, _, Hh, Wh = guide_hr.shape
    dev = feat_lr.device

    # 1. 引导图映射
    guide_luma_hr = guide_hr.mean(dim=1, keepdim=True)
    guide_luma_lr = F.interpolate(guide_luma_hr, (Hl, Wl), mode='bilinear', align_corners=False)

    # 2. Splatting (构建网格)
    bin_idx = torch.arange(num_bins, device=dev).view(1, 1, num_bins, 1, 1)
    dist = torch.abs((guide_luma_lr * (num_bins - 1)).unsqueeze(2) - bin_idx)
    weight = torch.clamp(1.0 - dist / (sr.unsqueeze(2) * num_bins + 1e-5), min=0)

    grid_num = feat_lr.unsqueeze(2) * weight 
    grid_den = weight 
    grid_all = torch.cat([grid_num, grid_den], dim=1) 

    # 3. 简化平滑 (各向异性耦合)
    grid_all = F.avg_pool3d(grid_all, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
    grid_all = grid_all * (1.0 + 0.01 * (sx + sy).unsqueeze(2))

    # 4. Slicing (采样)
    y_hr = torch.linspace(-1, 1, Hh, device=dev)
    x_hr = torch.linspace(-1, 1, Wh, device=dev)
    z_hr = (guide_luma_hr.squeeze(1) * 2.0 - 1.0) 

    grid_y, grid_x = torch.meshgrid(y_hr, x_hr, indexing='ij')
    sampling_coords = torch.stack([grid_x.expand(B, -1, -1), grid_y.expand(B, -1, -1), z_hr], dim=-1)

    sliced = F.grid_sample(grid_all, sampling_coords.unsqueeze(1), mode='bilinear', padding_mode='border', align_corners=True)
    sliced = sliced.squeeze(2)

    return sliced[:, :C] / sliced[:, C:].clamp_min(1e-8)

# --- 3. 联合优化模型 ---

class UPA_FBS_JointModel(nn.Module):
    def __init__(self, Hl, Wl, scale=16, init_sigma=8.0, init_sigma_r=0.1, num_bins=12):
        super().__init__()
        # UPA 部分：负责跨分辨率扩张
        self.sx_raw = nn.Parameter(torch.full((1, 1, Hl, Wl), float(np.log(init_sigma))))
        self.sy_raw = nn.Parameter(torch.full((1, 1, Hl, Wl), float(np.log(init_sigma))))
        self.th_raw = nn.Parameter(torch.zeros((1, 1, Hl, Wl)))
        self.sr_raw = nn.Parameter(torch.full((1, 1, Hl, Wl), float(np.log(init_sigma_r))))
        self.num_bins = num_bins

        # FBS 部分：负责像素级提纯 (lam 调大可增强平滑感)
        self.fbs = DifferentiableFBS(lam=64.0, sigma_luma=0.1, iters=3)

    def forward(self, feat_lr, guide_hr):
        sx = torch.exp(self.sx_raw)
        sy = torch.exp(self.sy_raw)
        th = _tanh_bound_pi(self.th_raw)
        sr = torch.exp(self.sr_raw)

        # 1. 基础 UPA 上采样
        upa_out = gs_jbu_grid_differentiable(feat_lr, guide_hr, sx, sy, th, sr, num_bins=self.num_bins)
        
        # 2. 联合 FBS 提纯 (梯度可穿透)
        final_out = self.fbs(upa_out, guide_hr)
        
        return final_out

# --- 4. 启动函数 ---

def UPA_Joint_FBS(HR_img, lr_modality):
    """
    联合 UPA-FBS 优化器
    """
    with torch.enable_grad():
        start_time = time.time()
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_type)
        
        # 数据转换
        if isinstance(HR_img, torch.Tensor):
            hr = HR_img.detach().cpu()
            if hr.dim() == 3: hr = hr.unsqueeze(0)
        else:
            hr = torch.from_numpy(np.array(HR_img)).permute(2, 0, 1).unsqueeze(0).float()
        
        hr = hr.to(device) / 255.0
        H, W = hr.shape[-2:]
        
        if isinstance(lr_modality, torch.Tensor):
            lr_modality = lr_modality.to(device)
        else:
            lr_modality = torch.from_numpy(np.array(lr_modality)).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
        
        Hl, Wl = lr_modality.shape[-2:]
        scale = int(H / Hl)
        
        # 构造训练目标
        lr_train_input = F.interpolate(hr, scale_factor=1/scale, mode="bicubic", align_corners=False)

        # 初始化联合模型
        model = UPA_FBS_JointModel(Hl, Wl, scale=scale).to(device)
        model.train()

        opt = torch.optim.Adam(model.parameters(), lr=1e-1)
        max_steps = 25 # 略微增加步骤，因为 FBS 引入了更多约束
        scheduler = LambdaLR(opt, lr_lambda=lambda step: 0.95 ** step)
        scaler = torch.amp.GradScaler(device_type, enabled=True)

        print(f"\n[UPA-FBS 联合优化] 启动 | 尺寸: {H}x{W} | 优化级: 像素级+网格级")

        for step in range(1, max_steps + 1):
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type, enabled=True, dtype=torch.float16):
                # 这里预测出的 pred 已经经过了 FBS 的提纯
                pred = model(lr_train_input, hr) 
                loss = F.l1_loss(pred, hr)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()

            if step % 5 == 0:
                print(f"  > Step {step:2d} | Loss: {loss.item():.6f}")

        model.eval()
        with torch.no_grad():
            final_feat = model(lr_modality.to(torch.float32), hr)
        
        print(f"[UPA-FBS] 完成 | 总耗时: {time.time() - start_time:.2f}s")
        return final_feat