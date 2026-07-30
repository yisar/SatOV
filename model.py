import os
from typing import Union, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from dino_splat_ov.gsup import GaussianUpsamplerWrapper
from satup.model import SatUp

ROOT = os.path.dirname(os.path.abspath(__file__))



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
        self.classnames = classnames if classnames is not None else ["object"]
        self.templates = templates if templates is not None else ["a photo of a {}."]
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)

        # 6. 初始化零样本分类器
        self._init_zeroshot_classifier()
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

        # 3. 最后一层 Self-Self Attention (ClearCLIP)
        last_block = blocks[-1]
        x_norm = last_block.ln_1(x_tokens)
        attn = last_block.attn

        qkv = F.linear(x_norm, attn.in_proj_weight, attn.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        B, N, D = q.shape
        num_heads = attn.num_heads
        head_dim = D // num_heads

        q = q.view(B, N, num_heads, head_dim).transpose(1, 2)
        v = v.view(B, N, num_heads, head_dim).transpose(1, 2)

        attn_matrix = (q @ q.transpose(-2, -1)) * (head_dim ** -0.5)
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