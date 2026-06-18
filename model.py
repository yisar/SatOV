import os
from typing import Union, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from aaa import GaussianUpsamplerWrapper
from grid_jbu import GridJBU
from bench.segearthov import load_featup_upsampler
from jafar import SatUp

ROOT = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CLASSNAMES = ["object"]
_DEFAULT_TEMPLATES = ["a photo of a {}."]


class DenseClip(nn.Module):
    def __init__(
        self,
        name: str,
        classnames: List[str] = None,
        templates: List[str] = None,
        device: Union[str, torch.device] = "cuda"
        if torch.cuda.is_available()
        else "cpu",
        jit: bool = False,
        only_clear: bool = False,
        upsampler="gfup",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.model_name = name
        self.only_clear = only_clear

        # 1. 加载 OpenCLIP 模型
        pretrained_tag = "openai" if "laion" not in name else "laion2b_s34b_b88k"
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name=name, pretrained=pretrained_tag, device=str(self.device), jit=jit
        )
        self.clip_model = model.to(self.device)
        self.preprocess = preprocess
        self.visual = self.clip_model.visual

        # 2. 维度定义
        # ViT-B-16 的 conv1.out_channels 通常为 768
        self.feat_dim = self.visual.conv1.out_channels
        self.embed_dim = (
            model.text_projection.shape[1]
            if hasattr(model, "text_projection")
            else self.feat_dim
        )
        self.up = None

        if upsampler == "anyup":
            self.up = (
                torch.hub.load("wimmerth/anyup", "anyup", verbose=False)
                .to(self.device)
                .eval()
            )
        elif upsampler == "satup":
            self.up = SatUp()
        elif upsampler == "gfup":
            self.up = GaussianUpsamplerWrapper()
        elif upsampler == "featup":
            hub_model = load_featup_upsampler(device=self.device)
            self.up = lambda g, f: hub_model(g, f)

        if self.only_clear is not False:
            self.up = None

        # 将 CLIP 原生的视觉投影权重迁移到 Conv2d(1x1) 中，方便处理特征图
        self.v_proj = nn.Conv2d(self.feat_dim, self.embed_dim, 1).to(self.device)
        if hasattr(self.visual, "proj") and self.visual.proj is not None:
            with torch.no_grad():
                # 权重转置并扩展为 [out, in, 1, 1]
                self.v_proj.weight.data.copy_(
                    self.visual.proj.data.T.unsqueeze(-1).unsqueeze(-1)
                )
                if self.v_proj.bias is not None:
                    nn.init.constant_(self.v_proj.bias, 0)

        # 5. 文本分支配置
        self.classnames = classnames if classnames is not None else _DEFAULT_CLASSNAMES
        self.templates = templates if templates is not None else _DEFAULT_TEMPLATES
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)

        # 6. 初始化零样本分类器 (支持近义词)
        self._init_zeroshot_classifier()

    @torch.no_grad()
    def _init_zeroshot_classifier(self):
        """
        核心逻辑：对每个类别组内的所有近义词进行编码，取均值后归一化。
        """
        final_text_embeds = []
        for class_group in self.classnames:
            synonyms = [s.strip() for s in class_group.split(",")]
            group_embeds = []
            for cls_name in synonyms:
                texts = [t.format(cls_name) for t in self.templates]
                tokens = open_clip.tokenize(texts).to(self.device)

                # [num_templates, embed_dim]
                class_embed = self.clip_model.encode_text(tokens)
                class_embed = F.normalize(class_embed, dim=-1)
                group_embeds.append(class_embed.mean(dim=0))

            combined_embed = torch.stack(group_embeds, dim=0).mean(dim=0)
            combined_embed = F.normalize(combined_embed, dim=-1)
            final_text_embeds.append(combined_embed)

        weights = torch.stack(final_text_embeds, dim=1).to(self.device)
        self.zeroshot_weights = nn.Parameter(weights)

    def _stem(self, x, hr_guide: Optional[torch.Tensor] = None):
        B, C, H, W = x.shape

        # --- 1. CLIP 视觉预处理与位置编码插值 ---
        x_in = self.visual.conv1(x)
        grid_h, grid_w = x_in.shape[2], x_in.shape[3]
        x_tokens = x_in.flatten(2).permute(0, 2, 1)  # [B, HW, D]

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

        # --- 2. Transformer 前馈 (保留前 L-1 层) ---
        blocks = self.visual.transformer.resblocks
        for i in range(len(blocks) - 1):
            x_tokens = blocks[i](x_tokens)

        # --- 3. ClearCLIP 核心：最后一层 Self-Self Attention ---
        last_block = blocks[-1]
        x_norm = last_block.ln_1(x_tokens)
        attn = last_block.attn

        # 提取 QKV 矩阵
        qkv = F.linear(x_norm, attn.in_proj_weight, attn.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        B, N, D = q.shape
        num_heads = attn.num_heads
        head_dim = D // num_heads

        q = q.view(B, N, num_heads, head_dim).transpose(1, 2)
        v = v.view(B, N, num_heads, head_dim).transpose(1, 2)

        # Self-Self Attention 计算 (Patch 间关系，不依赖全局 K)
        attn_matrix = (q @ q.transpose(-2, -1)) * (head_dim**-0.5)
        attn_matrix = attn_matrix.softmax(dim=-1)
        attn_out = (attn_matrix @ v).transpose(1, 2).reshape(B, N, -1)

        # 投影与后归一化
        x_feat = F.linear(attn_out, attn.out_proj.weight, attn.out_proj.bias)
        x_feat = self.visual.ln_post(x_feat)  # [B, N, D]

        # --- 4. 消除 CLS 全局偏置 (De-biasing) ---
        # 逻辑：Patch Tokens = Patch Tokens - CLS Token
        cls_token_out = x_feat[:, :1, :]  # [B, 1, D]
        patch_tokens_out = x_feat[:, 1:, :]  # [B, HW, D]

        # 利用广播机制，让每个 patch 减去该图像对应的全局背景均值
        debiased_patches = patch_tokens_out - cls_token_out

        # 将特征还原为 2D 形状
        lr_features = debiased_patches.permute(0, 2, 1).reshape(
            B, self.feat_dim, grid_h, grid_w
        )

        # --- 5. 引导上采样 ---
        guide = hr_guide if hr_guide is not None else x
        if self.up is not None:
            up_features = self.up(guide, lr_features)
        else:
            up_features = F.interpolate(
                lr_features, size=(H, W), mode="bilinear", align_corners=False
            )

        return up_features

    def forward(self, images, hr_guide: Optional[torch.Tensor] = None):
        """
        输入: images [B, 3, H, W]
        输出: logits [B, num_classes, H, W]
        """
        # 1. 提取密集特征 (已去偏置并上采样)
        features = self._stem(
            images.to(self.device),
            hr_guide.to(self.device) if hr_guide is not None else None,
        )

        # 🔥 最小修复：只在 mismatch 时重建 v_proj
        # if features.shape[1] != self.v_proj.in_channels:
        # self.v_proj = nn.Conv2d(features.shape[1], self.embed_dim, 1).to(self.device)

        # 2. 视觉投影到语义嵌入空间并归一化
        # features: [B, feat_dim, H, W] -> [B, embed_dim, H, W]
        features = self.v_proj(features)
        features = F.normalize(features, dim=1)

        B, C, H_f, W_f = features.shape

        # 3. 计算与文本权重的相似度
        # 将特征展平进行矩阵乘法: [B*H*W, C] @ [C, num_classes]
        flat_features = features.permute(0, 2, 3, 1).reshape(-1, C)
        logits = (flat_features @ self.zeroshot_weights) / self.temperature

        # 4. 还原形状为 [B, num_classes, H, W]
        return logits.reshape(B, H_f, W_f, -1).permute(0, 3, 1, 2)
