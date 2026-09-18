import os
from typing import Union, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from dino_splat_ov.gsup import GaussianUpsamplerWrapper
from satup.model import SatUp

ROOT = os.path.dirname(os.path.abspath(__file__))


class CLIPResQQ(nn.Module):
    def __init__(
        self,
        name: str,
        classnames: List[str] = None,
        templates: List[str] = None,
        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
        jit: bool = False,
        upsampler: str = "gsup",
        # ResCLIP 相关参数
        use_resclip: bool = True,
        resclip_alpha: float = 0.5,  # 残差融合权重
        resclip_layer: int = -2,     # 提取中间层 (-2 表示倒数第二层)
    ):
        super().__init__()
        self.device = torch.device(device)
        self.model_name = name
        self.use_resclip = use_resclip
        self.resclip_alpha = resclip_alpha
        self.resclip_layer = resclip_layer

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
            self.up = SatUp(dim=128, v_dim=768).to(self.device)
            ckpt = torch.load("satup.pth", map_location=self.device)
            self.up.load_state_dict(ckpt, strict=True)
            self.up.eval()
        elif upsampler == "gsup":
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
        self.classnames = classnames if classnames is not None else ["object"]
        self.templates = templates if templates is not None else ["a photo of a {}."]
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)

        # 6. 初始化零样本分类器
        self._init_zeroshot_classifier()

    def extract_patch_features(self, images: torch.Tensor):
        """
        只提取 ClearCLIP 去偏后的 Patch 级特征，不做任何上采样和投影。
        返回: [B, C, grid_h, grid_w]
        """
        images = images.to(self.device)
        lr_features = self._extract_clearclip_features(images)
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

            # --- 方案二：两两相似度 softmax 加权平均 ---
            group_embeds = torch.stack(group_embeds, dim=0)          # [K, D]
            group_embeds = F.normalize(group_embeds, dim=-1)
            sim = group_embeds @ group_embeds.T                      # [K, K]
            scores = sim.sum(dim=1)                                  # [K]
            weights = F.softmax(scores / 0.07, dim=0)                # [K]
            combined_embed = (weights.unsqueeze(-1) * group_embeds).sum(dim=0)
            combined_embed = F.normalize(combined_embed, dim=-1)
            final_text_embeds.append(combined_embed)

        weights = torch.stack(final_text_embeds, dim=1).to(self.device)
        self.zeroshot_weights = nn.Parameter(weights)

    def _extract_clearclip_features(self, img: torch.Tensor):
        """
        从输入图像提取特征。如果启用 ResCLIP，则在最后一层融合中间层的互相关注意力。
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

        # 2. Transformer 层
        blocks = self.visual.transformer.resblocks
        num_layers = len(blocks)

        # 用于存储中间层的互相关注意力 (用于 ResCLIP)
        intermediate_attn = None

        # 2.1 前向传播至倒数第二层，并提取中间层注意力
        for i in range(num_layers - 1):
            # 对于倒数第二层，我们需要提取其 Query-Key 注意力
            if self.use_resclip and i == num_layers + self.resclip_layer:
                # 手动计算该层的 Query-Key 注意力
                block = blocks[i]
                # 获取该层的 Q 和 K
                # 注意：这里需要进入 block 内部计算，但为了不破坏原有结构，我们复制一份计算逻辑
                # 更简洁的方式是直接使用 block 的 forward，但我们需要的是注意力矩阵，而不是输出
                # 这里采用与原始 CLIP 一致的计算方式
                x_norm_mid = block.ln_1(x_tokens)
                attn_mid = block.attn
                qkv_mid = F.linear(x_norm_mid, attn_mid.in_proj_weight, attn_mid.in_proj_bias)
                q_mid, k_mid, _ = qkv_mid.chunk(3, dim=-1)

                B_mid, N_mid, D_mid = q_mid.shape
                num_heads_mid = attn_mid.num_heads
                head_dim_mid = D_mid // num_heads_mid

                q_mid = q_mid.view(B_mid, N_mid, num_heads_mid, head_dim_mid).transpose(1, 2)
                k_mid = k_mid.view(B_mid, N_mid, num_heads_mid, head_dim_mid).transpose(1, 2)

                # 计算 Query-Key 互相关注意力 (非最终层具有空间定位能力)
                mid_attn_matrix = (q_mid @ k_mid.transpose(-2, -1)) * (head_dim_mid ** -0.5)
                intermediate_attn = mid_attn_matrix.softmax(dim=-1)

                # 继续正常的前向传播
                x_tokens = block(x_tokens)
            else:
                x_tokens = blocks[i](x_tokens)

        # 3. 最后一层 Self-Self Attention (ClearCLIP) + ResCLIP 融合
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

        # 3.1 计算原始的 Query-Query 自注意力 (ClearCLIP)
        attn_matrix = (q @ q.transpose(-2, -1)) * (head_dim ** -0.5)
        attn_matrix_qq = attn_matrix.softmax(dim=-1)

        # 3.2 如果启用 ResCLIP，进行残差融合
        if self.use_resclip and intermediate_attn is not None:
            # 中间层注意力是 [B, heads_mid, N, N]，需要确保与当前注意力形状一致
            # 如果中间层的 head 数量不同，需要进行平均或投影
            if intermediate_attn.shape[1] != num_heads:
                # 如果 heads 数量不同，在 head 维度进行平均
                intermediate_attn = intermediate_attn.mean(dim=1, keepdim=True)
                intermediate_attn = intermediate_attn.expand(-1, num_heads, -1, -1)

            # 将中间层注意力移到与当前张量相同的设备和数据类型
            intermediate_attn = intermediate_attn.to(device=attn_matrix_qq.device, dtype=attn_matrix_qq.dtype)

            # 残差融合: 新的注意力 = (1 - alpha) * 最后层注意力 + alpha * 中间层注意力
            # 注意：这里使用残差连接的思想，融合两种注意力
            attn_matrix_fused = (1 - self.resclip_alpha) * attn_matrix_qq + self.resclip_alpha * intermediate_attn
            # 重新归一化
            attn_matrix_fused = attn_matrix_fused / (attn_matrix_fused.sum(dim=-1, keepdim=True) + 1e-8)

            # 使用融合后的注意力对 Value 进行加权
            attn_out = (attn_matrix_fused @ v).transpose(1, 2).reshape(B, N, -1)
        else:
            # 原始 ClearCLIP 路径
            attn_out = (attn_matrix_qq @ v).transpose(1, 2).reshape(B, N, -1)

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

        # --- 针对 SatUp 的特殊处理 ---
        if isinstance(self.up, SatUp):
            lr_img = x
            lr_features = self._extract_clearclip_features(lr_img)
            up_features = self.up(lr_img, lr_features, output_size=(H, W))
            return up_features

        # 通用流程
        lr_features = self._extract_clearclip_features(x)
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