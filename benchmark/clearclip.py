import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from PIL import Image
import numpy as np
from typing import Union, List

# ===================== 固定配置 =====================
TARGET_LABELS = [
    'background', 'bareland', 'pavement', 'road', 'water',
    'tree', 'grass', 'cropland', 'building'
]

CUSTOM_PALETTE = [
    (68, 1, 84), (72, 40, 120), (62, 74, 137), (49, 104, 142),
    (38, 130, 142), (31, 158, 137), (73, 193, 110), (160, 218, 57),
    (253, 231, 37)
]

_TEMPLATES = ['a satellite photo of a {}.']

# ===================== PAMR 辅助模块 =====================

class LocalAffinity(nn.Module):
    def __init__(self, dilations=[1]):
        super(LocalAffinity, self).__init__()
        self.dilations = dilations
        self.register_buffer('kernel', self._init_aff())

    def _init_aff(self):
        weight = torch.zeros(8, 1, 3, 3)
        for i in range(weight.size(0)):
            weight[i, 0, 1, 1] = 1
        weight[0, 0, 0, 0] = -1; weight[1, 0, 0, 1] = -1; weight[2, 0, 0, 2] = -1
        weight[3, 0, 1, 0] = -1; weight[4, 0, 1, 2] = -1
        weight[5, 0, 2, 0] = -1; weight[6, 0, 2, 1] = -1; weight[7, 0, 2, 2] = -1
        return weight

    def forward(self, x):
        B, K, H, W = x.size()
        x = x.view(B * K, 1, H, W)
        x_affs = []
        for d in self.dilations:
            x_pad = F.pad(x, [d] * 4, mode='replicate')
            x_aff = F.conv2d(x_pad, self.kernel, dilation=d)
            x_affs.append(x_aff)
        x_aff = torch.cat(x_affs, 1)
        return x_aff.view(B, K, -1, H, W)

class LocalAffinityCopy(LocalAffinity):
    def _init_aff(self):
        weight = torch.zeros(8, 1, 3, 3)
        indices = [(0,0), (0,1), (0,2), (1,0), (1,2), (2,0), (2,1), (2,2)]
        for i, (r, c) in enumerate(indices):
            weight[i, 0, r, c] = 1
        return weight

class LocalStDev(LocalAffinity):
    def _init_aff(self):
        return torch.ones(9, 1, 3, 3)
    def forward(self, x):
        x = super().forward(x)
        return x.std(2, keepdim=True)

class PAMR(nn.Module):
    def __init__(self, num_iter=10, dilations=[1, 2, 4, 8, 12]):
        super(PAMR, self).__init__()
        self.num_iter = num_iter
        self.aff_x = LocalAffinity(dilations)
        self.aff_m = LocalAffinityCopy(dilations)
        self.aff_std = LocalStDev(dilations)

    def forward(self, x, mask):
        if mask.shape[-2:] != x.shape[-2:]:
            mask = F.interpolate(mask, size=x.shape[-2:], mode="bilinear", align_corners=True)
        
        B, C, H, W = mask.size()
        x_std = self.aff_std(x)
        aff = -torch.abs(self.aff_x(x)) / (1e-8 + 0.1 * x_std)
        aff = aff.mean(1, keepdim=True) 
        aff = F.softmax(aff, dim=2)

        for _ in range(self.num_iter):
            m = self.aff_m(mask)
            mask = (m * aff).sum(2)
        return mask

# ===================== DenseClip + PAMR 主类 =====================

class DenseClipPAMR(nn.Module):
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.device = torch.device(device)
        model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-16", pretrained='openai', device=self.device)
        self.clip_model = model
        self.preprocess = preprocess
        self.visual = model.visual
        
        self.feat_dim = 768
        self.embed_dim = model.text_projection.shape[1]
        self.v_proj = nn.Conv2d(self.feat_dim, self.embed_dim, 1).to(self.device)
        if hasattr(self.visual, 'proj'):
            with torch.no_grad():
                self.v_proj.weight.copy_(self.visual.proj.T.unsqueeze(-1).unsqueeze(-1))

        self.temperature = nn.Parameter(torch.ones([]) * 0.07)
        self.pamr = PAMR(num_iter=10).to(self.device)
        self._init_zeroshot_classifier()
        self.eval()

    @torch.no_grad()
    def _init_zeroshot_classifier(self):
        text_embeds = []
        for cls_name in TARGET_LABELS:
            texts = [_TEMPLATES[0].format(cls_name)]
            tokens = open_clip.tokenize(texts).to(self.device)
            embed = F.normalize(self.clip_model.encode_text(tokens).mean(dim=0), dim=-1)
            text_embeds.append(embed)
        self.zeroshot_weights = nn.Parameter(torch.stack(text_embeds, dim=1))

    def _extract_dense_features(self, x):
        B, C, H, W = x.shape
        x_in = self.visual.conv1(x)
        h, w = x_in.shape[2], x_in.shape[3]
        x_tokens = x_in.flatten(2).permute(0, 2, 1)
        pos_embed = self.visual.positional_embedding
        cls_pos = pos_embed[:1, :]
        patch_pos = pos_embed[1:, :].reshape(1, 14, 14, -1).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(patch_pos, size=(h, w), mode='bicubic', align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(h * w, -1)
        x_tokens = torch.cat([self.visual.class_embedding.expand(B, 1, -1), x_tokens], dim=1)
        x_tokens = x_tokens + torch.cat([cls_pos, patch_pos], dim=0)
        x_tokens = self.visual.ln_pre(x_tokens)

        for i, block in enumerate(self.visual.transformer.resblocks):
            if i == len(self.visual.transformer.resblocks) - 1:
                x_norm = block.ln_1(x_tokens)
                qkv = F.linear(x_norm, block.attn.in_proj_weight, block.attn.in_proj_bias)
                _, _, v = qkv.chunk(3, dim=-1)
                x_tokens = x_tokens + block.attn.out_proj(v) 
                x_tokens = block.ln_2(x_tokens)
                x_tokens = x_tokens + block.mlp(x_tokens)
            else:
                x_tokens = block(x_tokens)

        x_feat = self.visual.ln_post(x_tokens[:, 1:, :])
        return x_feat.permute(0, 2, 1).reshape(B, self.feat_dim, h, w)

    @torch.no_grad()
    def forward(self, image_path: str, max_width: int = 800):
        # 1. 自适应尺寸缩放逻辑
        raw_img = Image.open(image_path).convert("RGB")
        orig_w, orig_h = raw_img.size
        
        if orig_w > max_width:
            scale = max_width / orig_w
            new_w, new_h = max_width, int(orig_h * scale)
            proc_img = raw_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        else:
            new_w, new_h = orig_w, orig_h
            proc_img = raw_img

        # 2. 正常推理流程
        img_tensor = self.preprocess(proc_img).unsqueeze(0).to(self.device)
        feat_map = self._extract_dense_features(img_tensor)
        feat_map = F.normalize(self.v_proj(feat_map), dim=1)
        logits = (feat_map.permute(0, 2, 3, 1) @ self.zeroshot_weights) / self.temperature
        logits = logits.permute(0, 3, 1, 2)

        # 3. PAMR 细化
        # 使用缩放后的 img_tensor 尺寸作为引导图
        guide_img = F.interpolate(img_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
        refined_mask = self.pamr(guide_img, logits)
        
        # 4. 后处理与尺寸还原
        pred_mask = refined_mask.argmax(dim=1).squeeze(0).cpu().numpy()
        color_output = np.zeros((new_h, new_w, 3), dtype=np.uint8)
        for idx, color in enumerate(CUSTOM_PALETTE):
            color_output[pred_mask == idx] = color
            
        final_result = Image.fromarray(color_output)
        if (new_w, new_h) != (orig_w, orig_h):
            final_result = final_result.resize((orig_w, orig_h), Image.Resampling.NEAREST)
            
        return final_result

# ===================== 推理 =====================
if __name__ == "__main__":
    import os
    IMAGE_PATH = "img3.jpg"
    OUT_PATH = "./img2/clearclip.png"
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    
    if os.path.exists(IMAGE_PATH):
        model = DenseClipPAMR()
        print(f"正在处理: {IMAGE_PATH} (限制宽度: 512)...")
        # 如果显存依然不够，可以尝试将 max_width 设为 448 或更小
        result = model(IMAGE_PATH, max_width=800)
        result.save(OUT_PATH)
        print(f"完成！结果已保存至: {OUT_PATH}")
    else:
        print("图片不存在。")