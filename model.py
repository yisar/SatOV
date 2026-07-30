import os
from typing import Union, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from dino_splat_ov.gsup import GaussianUpsamplerWrapper
from satup.model import SatUp

ROOT = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CLASSNAMES = ["object"]
_DEFAULT_TEMPLATES = ["a photo of a {}."]


class DenseClip(nn.Module):
    def __init__(
        self,
        name: str,
        classnames: List[str] = None,
        templates: List[str] = None,
        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
        jit: bool = False,
        only_clear: bool = False,
        upsampler: str = "gfup",
        use_nap: bool = True,            # <-- 新增：是否启用 NAP 空间偏置
        pos_bias_scale: float = 0.1,     # <-- 新增：空间偏置的缩放系数
    ):
        super().__init__()
        self.device = torch.device(device)
        self.model_name = name
        self.only_clear = only_clear
        self.use_nap = use_nap            # 保存参数
        self.pos_bias_scale = pos_bias_scale

        # 1. 加载 OpenCLIP 模型
        pretrained_tag = "openai" if "laion" not in name else "laion2b_s34b_b88k"
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name=name, pretrained=pretrained_tag, device=str(self.device), jit=jit
        )
        self.clip_model = model.to(self.device)
        self.preprocess = preprocess
        self.visual = self.clip_model.visual

        # 2. 维度定义
        self.feat_dim = self.visual.conv1.out_channels
        self.embed_dim = (
            model.text_projection.shape[1]
            if hasattr(model, "text_projection")
            else self.feat_dim
        )
        self.up = None

        # 3. 上采样器初始化
        if upsampler == "anyup":
            self.up = (
                torch.hub.load("wimmerth/anyup", "anyup", verbose=False)
                .to(self.device)
                .eval()
            )
        elif upsampler == "satup":
            # 直接使用 SatUp，不再通过 Wrapper，确保与 infer.py 调用方式一致
            self.up = SatUp(dim=128, v_dim=768).to(self.device)
            ckpt = torch.load("satup.pth", map_location=self.device)
            self.up.load_state_dict(ckpt, strict=True)
            self.up.eval()
        elif upsampler == "gsu":
            self.up = GaussianUpsamplerWrapper()

        # 4. 视觉投影层 (1x1 conv)
        self.v_proj = nn.Conv2d(self.feat_dim, self.embed_dim, 1).to(self.device)
        if hasattr(self.visual, "proj") and self.visual.proj is not None:
            with torch.no_grad():
                self.v_proj.weight.data.copy_(
                    self.visual.proj.data.T.unsqueeze(-1).unsqueeze(-1)
                )
                if self.v_proj.bias is not None:
                    nn.init.constant_(self.v_proj.bias, 0)

        # 5. 文本分支
        self.classnames = classnames if classnames is not None else _DEFAULT_CLASSNAMES
        self.templates = templates if templates is not None else _DEFAULT_TEMPLATES
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)

        # 6. 初始化零样本分类器
        self._init_zeroshot_classifier()

        # 7. 新增：位置偏置缓存（用于 NAP）   <-- 新增
        self._pos_bias_cache = {}

    # ------------- 新增 NAP 辅助方法 -------------
    @staticmethod
    def create_spatial_kernel(height, width, std=1.0, kernel_type='gaussian'):
        """生成空间核矩阵（高斯或拉普拉斯）"""
        center_h, center_w = (height - 1) / 2.0, (width - 1) / 2.0
        y_coords = torch.arange(height, dtype=torch.float) - center_h
        x_coords = torch.arange(width, dtype=torch.float) - center_w
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        if kernel_type == 'gaussian':
            dist_sq = (x_grid**2 + y_grid**2) / (2 * std**2)
            return torch.exp(-dist_sq)
        elif kernel_type == 'laplacian':
            dist_l1 = torch.abs(x_grid) + torch.abs(y_grid)
            return torch.exp(-dist_l1 / std)
        else:
            raise ValueError(f"不支持的核类型: {kernel_type}")

    @staticmethod
    def build_positional_attention_bias(patch_h, patch_w, spatial_kernel, adjust_for_cls=True):
        """利用空间核构建位置注意力偏置矩阵"""
        total_patches = patch_h * patch_w
        identity = torch.eye(total_patches).view(total_patches, patch_h, patch_w)
        convolved = F.conv2d(
            identity.unsqueeze(1),
            spatial_kernel.unsqueeze(0).unsqueeze(1),
            padding='same'
        ).squeeze(1)
        attn_bias = convolved.view(total_patches, total_patches)
        if adjust_for_cls:
            full_bias = torch.zeros((total_patches + 1, total_patches + 1))
            full_bias[1:, 1:] = attn_bias
            return full_bias
        return attn_bias
    # ---------------------------------------------

    def extract_patch_features(self, images: torch.Tensor):
        """
        只提取 ClearCLIP 去偏后的 Patch 级特征，不做任何上采样和投影。
        返回: [B, C, grid_h, grid_w]  例如 [B, 768, 14, 14]
        """
        images = images.to(self.device)
        lr_features = self._extract_clearclip_features(images)  # [B, C, grid_h, grid_w]
        return lr_features

    @torch.no_grad()
    def _init_zeroshot_classifier(self):
        final_text_embeds = []
        for class_group in self.classnames:
            synonyms = [s.strip() for s in class_group.split(",")]
            group_embeds = []
            for cls_name in synonyms:
                texts = [t.format(cls_name) for t in self.templates]
                tokens = open_clip.tokenize(texts).to(self.device)
                class_embed = self.clip_model.encode_text(tokens)
                class_embed = F.normalize(class_embed, dim=-1)
                group_embeds.append(class_embed.mean(dim=0))
            combined_embed = torch.stack(group_embeds, dim=0).mean(dim=0)
            combined_embed = F.normalize(combined_embed, dim=-1)
            final_text_embeds.append(combined_embed)
        weights = torch.stack(final_text_embeds, dim=1).to(self.device)
        self.zeroshot_weights = nn.Parameter(weights)

    def _extract_clearclip_features(self, img: torch.Tensor):
        """
        从输入图像提取 ClearCLIP 去偏特征 (patch - cls)
        如果 use_nap=True，在 QQ 注意力基础上叠加空间偏置。
        返回: [B, feat_dim, grid_h, grid_w]
        """
        B, C, H, W = img.shape

        # 1. conv1 + 位置编码插值
        x = self.visual.conv1(img)
        grid_h, grid_w = x.shape[2], x.shape[3]
        x_tokens = x.flatten(2).permute(0, 2, 1)  # [B, HW, D]

        cls_token = self.visual.class_embedding.to(x_tokens.dtype)
        pos_embed = self.visual.positional_embedding.to(x_tokens.dtype)

        cls_pos = pos_embed[:1, :]
        patch_pos = pos_embed[1:, :]
        old_grid = int(patch_pos.shape[0] ** 0.5)

        if old_grid != grid_h or old_grid != grid_w:
            patch_pos = patch_pos.reshape(1, old_grid, old_grid, -1).permute(0, 3, 1, 2)
            patch_pos = F.interpolate(
                patch_pos, size=(grid_h, grid_w), mode="bicubic", align_corners=False
            )
            patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(grid_h * grid_w, -1)
            new_pos_embed = torch.cat([cls_pos, patch_pos], dim=0)
        else:
            new_pos_embed = pos_embed

        x_tokens = torch.cat([cls_token.expand(B, 1, -1), x_tokens], dim=1)
        x_tokens = x_tokens + new_pos_embed
        x_tokens = self.visual.ln_pre(x_tokens)

        # 2. Transformer 前 L-1 层
        blocks = self.visual.transformer.resblocks
        for i in range(len(blocks) - 1):
            x_tokens = blocks[i](x_tokens)

        # 3. 最后一层 Self-Self Attention (ClearCLIP + NAP)
        last_block = blocks[-1]
        x_norm = last_block.ln_1(x_tokens)
        attn = last_block.attn

        qkv = F.linear(x_norm, attn.in_proj_weight, attn.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        B, N, D = q.shape
        num_heads = attn.num_heads
        head_dim = D // num_heads

        q = q.view(B, N, num_heads, head_dim).transpose(1, 2)  # [B, heads, N, hd]
        v = v.view(B, N, num_heads, head_dim).transpose(1, 2)

        # ---- 计算 QQ 注意力分数 ----
        attn_matrix = (q @ q.transpose(-2, -1)) * (head_dim ** -0.5)  # [B, heads, N, N]

        # ---- 【NAP 核心】叠加空间偏置 ----
        if self.use_nap:
            # ① 获取/生成空间偏置矩阵
            cache_key = (grid_h, grid_w)
            if cache_key not in self._pos_bias_cache:
                window_h, window_w = grid_h * 2 - 1, grid_w * 2 - 1
                # 混合高斯+拉普拉斯（权重可调，此处固定 0.7/0.3）
                gauss_k = self.create_spatial_kernel(window_h, window_w, std=1.0, kernel_type='gaussian')
                lapl_k = self.create_spatial_kernel(window_h, window_w, std=1.0, kernel_type='laplacian')
                mixed_kernel = 0.7 * gauss_k + 0.3 * lapl_k
                pos_bias = self.build_positional_attention_bias(grid_h, grid_w, mixed_kernel, adjust_for_cls=True)
                # 缓存到 CPU，节省显存（使用时再转至当前设备）
                self._pos_bias_cache[cache_key] = pos_bias.cpu()
            pos_bias = self._pos_bias_cache[cache_key].to(device=attn_matrix.device, dtype=attn_matrix.dtype)

            # ② 偏置加到注意力分数上（缩放系数可调）
            attn_matrix = attn_matrix + self.pos_bias_scale * pos_bias  # 广播到 [B, heads, N, N]

        # ---- Softmax 与加权 ----
        attn_matrix = attn_matrix.softmax(dim=-1)
        attn_out = (attn_matrix @ v).transpose(1, 2).reshape(B, N, -1)

        x_feat = F.linear(attn_out, attn.out_proj.weight, attn.out_proj.bias)
        x_feat = self.visual.ln_post(x_feat)

        # 4. De-biasing: patch = patch - cls
        cls_token_out = x_feat[:, :1, :]
        patch_tokens_out = x_feat[:, 1:, :]
        debiased_patches = patch_tokens_out - cls_token_out

        # 还原为 2D 特征图
        features = debiased_patches.permute(0, 2, 1).reshape(B, self.feat_dim, grid_h, grid_w)
        return features

    def _stem(self, x, hr_guide: Optional[torch.Tensor] = None):
        B, C, H, W = x.shape

        # --- 针对 SatUp 的特殊处理：与 infer.py 完全一致 ---
        if isinstance(self.up, SatUp):
            # 1. 下采样输入图像
            lr_img = x
            # 2. 从下采样图像提取 ClearCLIP 特征
            lr_features = self._extract_clearclip_features(lr_img)
            # 3. 调用 SatUp 上采样至原始尺寸
            up_features = self.up(lr_img, lr_features, output_size=(H, W))
            return up_features

        # 注意：原有逻辑是从高分辨率图像 x 提取特征，然后上采样
        lr_features = self._extract_clearclip_features(x)  # 高分辨率下的低分辨率特征图
        # guide = hr_guide if hr_guide is not None else x
        guide = x

        if self.up is not None:
            up_features = self.up(guide, lr_features)
        else:
            up_features = F.interpolate(lr_features, size=(H, W), mode='bilinear', align_corners=False)

        return up_features

    def forward(self, images, hr_guide: Optional[torch.Tensor] = None):
        """
        输入: images [B, 3, H, W]
        输出: logits [B, num_classes, H, W]
        """
        features = self._stem(images.to(self.device), hr_guide.to(self.device) if hr_guide is not None else None)

        # 投影到语义空间
        features = self.v_proj(features)
        features = F.normalize(features, dim=1)

        B, C, H_f, W_f = features.shape
        flat_features = features.permute(0, 2, 3, 1).reshape(-1, C)
        logits = (flat_features @ self.zeroshot_weights) / self.temperature
        return logits.reshape(B, H_f, W_f, -1).permute(0, 3, 1, 2)