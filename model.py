import json
import os
from typing import Union, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip

# 尝试加载 ROOT 路径
try:
    from libs.definitions import ROOT
except ImportError:
    ROOT = os.path.dirname(os.path.abspath(__file__))

_DEFAULT_CLASSNAMES = ["object"]

_DEFAULT_TEMPLATES = ['a photo of a {}.']

class DenseClip(nn.Module):
    def __init__(self, name: str, classnames: List[str] = None, templates: List[str] = None,
                 device: Union[str, torch.device] = 'cuda' if torch.cuda.is_available() else 'cpu',
                 jit: bool = False):
        super().__init__()
        self.device = torch.device(device)
        self.model_name = name

        # 1. 加载 OpenCLIP 模型
        pretrained_tag = 'openai' if 'laion' not in name else 'laion2b_s34b_b88k'
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name=name, pretrained=pretrained_tag, device=str(self.device), jit=jit
        )
        self.clip_model = model.to(self.device)
        self.preprocess = preprocess
        self.visual = self.clip_model.visual

        # 2. 维度定义
        self.feat_dim = self.visual.conv1.out_channels # ViT-B-16 为 768
        self.embed_dim = model.text_projection.shape[1] if hasattr(model, 'text_projection') else self.feat_dim

        # 3. 加载 AnyUp 引导上采样模块
        print(f"正在加载 AnyUp 预训练权重...")
        try:
            self.any_up = torch.hub.load("wimmerth/anyup", "anyup", verbose=False).to(self.device).eval()
        except Exception as e:
            print(f"警告：AnyUp 加载失败({e})，将回退至线性插值。")
            self.any_up = None

        # 4. 初始化视觉投影 (768 -> 512)
        self.v_proj = nn.Conv2d(self.feat_dim, self.embed_dim, 1).to(self.device)
        if hasattr(self.visual, 'proj') and self.visual.proj is not None:
            with torch.no_grad():
                self.v_proj.weight.data.copy_(self.visual.proj.data.T.unsqueeze(-1).unsqueeze(-1))
                if self.v_proj.bias is not None:
                    nn.init.constant_(self.v_proj.bias, 0)

        # 5. 文本分支
        self.classnames = classnames if classnames is not None else _DEFAULT_CLASSNAMES
        self.templates = templates if templates is not None else _DEFAULT_TEMPLATES
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)
        self._init_zeroshot_classifier()

    @torch.no_grad()
    def _init_zeroshot_classifier(self):
        text_embeds = []
        for cls_name in self.classnames:
            texts = [t.format(cls_name) for t in self.templates]
            tokens = open_clip.tokenize(texts).to(self.device)
            embed = self.clip_model.encode_text(tokens)
            embed = F.normalize(embed.mean(dim=0), dim=-1)
            text_embeds.append(embed)
        weights = torch.stack(text_embeds, dim=1).to(self.device)
        self.zeroshot_weights = nn.Parameter(F.normalize(weights, dim=0))

    def _stem(self, x, hr_guide: Optional[torch.Tensor] = None):
        B, C, H, W = x.shape
        
        # --- CLIP 视觉编码器 + 动态位置编码插值 ---
        x_in = self.visual.conv1(x) 
        grid_h, grid_w = x_in.shape[2], x_in.shape[3]
        x_tokens = x_in.flatten(2).permute(0, 2, 1)
        
        cls_token = self.visual.class_embedding.to(x_tokens.dtype)
        pos_embed = self.visual.positional_embedding.to(x_tokens.dtype)
        
        # 插值逻辑
        cls_pos = pos_embed[:1, :]
        patch_pos = pos_embed[1:, :]
        old_grid = int(patch_pos.shape[0]**0.5)
        if old_grid != grid_h or old_grid != grid_w:
            patch_pos = patch_pos.reshape(1, old_grid, old_grid, -1).permute(0, 3, 1, 2)
            patch_pos = F.interpolate(patch_pos, size=(grid_h, grid_w), mode='bicubic', align_corners=False)
            patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(grid_h * grid_w, -1)
            new_pos_embed = torch.cat([cls_pos, patch_pos], dim=0)
        else:
            new_pos_embed = pos_embed

        x_tokens = torch.cat([cls_token.expand(B, 1, -1), x_tokens], dim=1)
        x_tokens = x_tokens + new_pos_embed
        x_tokens = self.visual.ln_pre(x_tokens)

        # --- ClearCLIP 改造 ---
        blocks = self.visual.transformer.resblocks
        for i in range(len(blocks) - 1):
            x_tokens = blocks[i](x_tokens)
            
        last_block = blocks[-1]
        x_norm = last_block.ln_1(x_tokens)
        attn = last_block.attn
        
        qkv = F.linear(x_norm, attn.in_proj_weight, attn.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        B, N, _ = q.shape
        num_heads = attn.num_heads
        head_dim = _ // num_heads
        
        q = q.view(B, N, num_heads, head_dim).transpose(1, 2)
        v = v.view(B, N, num_heads, head_dim).transpose(1, 2)
        
        # Self-Self Attention (ClearCLIP)
        attn_matrix = (q @ q.transpose(-2, -1)) * (head_dim ** -0.5)
        attn_matrix = attn_matrix.softmax(dim=-1)
        attn_out = (attn_matrix @ v).transpose(1, 2).reshape(B, N, -1)
        
        x_feat = F.linear(attn_out, attn.out_proj.weight, attn.out_proj.bias)
        x_feat = self.visual.ln_post(x_feat)

        # --- AnyUp 引导上采样 ---
        lr_features = x_feat[:, 1:, :].permute(0, 2, 1).reshape(B, self.feat_dim, grid_h, grid_w)
        if self.any_up is not None:
            guide = hr_guide if hr_guide is not None else x
            up_features = self.any_up(guide, lr_features)
        else:
            up_features = F.interpolate(lr_features, size=(H, W), mode='bilinear', align_corners=False)
            
        return up_features

    def forward(self, images, hr_guide: Optional[torch.Tensor] = None):
        features = self._stem(images.to(self.device), hr_guide.to(self.device) if hr_guide is not None else None)
        features = F.normalize(self.v_proj(features), dim=1)
        B, C, H_f, W_f = features.shape
        logits = (features.permute(0, 2, 3, 1).reshape(-1, C) @ self.zeroshot_weights) / self.temperature
        return logits.reshape(B, H_f, W_f, -1).permute(0, 3, 1, 2)