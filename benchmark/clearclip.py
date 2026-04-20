import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from typing import List
import open_clip
import os

# ==========================================
# 1. PAMR 核心组件 (保持高效卷积实现)
# ==========================================

class LocalAffinity(nn.Module):
    def __init__(self, dilations=[1]):
        super(LocalAffinity, self).__init__()
        self.dilations = dilations
        self.register_buffer('kernel', self._init_aff())

    def _init_aff(self):
        weight = torch.zeros(8, 1, 3, 3)
        for i in range(weight.size(0)): weight[i, 0, 1, 1] = 1
        weight[0,0,0,0]=-1; weight[1,0,0,1]=-1; weight[2,0,0,2]=-1
        weight[3,0,1,0]=-1; weight[4,0,1,2]=-1
        weight[5,0,2,0]=-1; weight[6,0,2,1]=-1; weight[7,0,2,2]=-1
        return weight

    def forward(self, x):
        B, K, H, W = x.size()
        x = x.view(B * K, 1, H, W)
        x_affs = []
        for d in self.dilations:
            x_pad = F.pad(x, [d] * 4, mode='replicate')
            x_aff = F.conv2d(x_pad, self.kernel, dilation=d)
            x_affs.append(x_aff)
        return torch.cat(x_affs, 1).view(B, K, -1, H, W)

class LocalAffinityCopy(LocalAffinity):
    def _init_aff(self):
        weight = torch.zeros(8, 1, 3, 3)
        indices = [(0,0,0),(1,0,1),(2,0,2),(3,1,0),(4,1,2),(5,2,0),(6,2,1),(7,2,2)]
        for i, r, c in indices: weight[i, 0, r, c] = 1
        return weight

class LocalStDev(LocalAffinity):
    def _init_aff(self): return torch.ones(9, 1, 3, 3)
    def forward(self, x): return super(LocalStDev, self).forward(x).std(2, keepdim=True)

class LocalAffinityAbs(LocalAffinity):
    def forward(self, x): return torch.abs(super(LocalAffinityAbs, self).forward(x))

class PAMR(nn.Module):
    def __init__(self, num_iter=1, dilations=[1, 2, 4]):
        super(PAMR, self).__init__()
        self.num_iter = num_iter
        self.aff_x = LocalAffinityAbs(dilations)
        self.aff_m = LocalAffinityCopy(dilations)
        self.aff_std = LocalStDev(dilations)

    def forward(self, guide, mask):
        """
        guide: 原图 (B, 3, H, W)
        mask: 低分辨率 Logits (B, C, gh, gw)
        """
        # 将 mask 上采样到原图分辨率 (H, W)
        mask = F.interpolate(mask, size=guide.size()[-2:], mode="bilinear", align_corners=True)
        
        # 基于原图计算亲和性权重
        guide_std = self.aff_std(guide)
        guide_aff = -self.aff_x(guide) / (1e-8 + 0.1 * guide_std)
        # 在通道维度平滑权重，并进行 Softmax 归一化
        guide_aff = F.softmax(guide_aff.mean(1, keepdim=True), 2)
        
        # 迭代扩散传播
        for _ in range(self.num_iter):
            m = self.aff_m(mask)
            mask = (m * guide_aff).sum(2)
        return mask

# ==========================================
# 2. DenseClip 模型主体
# ==========================================

class DenseClip(nn.Module):
    def __init__(self, name: str, classnames: List[str], device='cuda'):
        super().__init__()
        self.device = torch.device(device)
        
        model, _, _ = open_clip.create_model_and_transforms(name, pretrained='openai', device=self.device)
        self.clip_model = model
        self.visual = model.visual
        self.feat_dim = self.visual.conv1.out_channels
        self.embed_dim = model.text_projection.shape[1] if hasattr(model, 'text_projection') else self.feat_dim

        # 增加 dilation 范围以适应高分辨率引导
        self.pamr = PAMR(num_iter=20, dilations=[1, 2, 4, 8]).to(self.device)

        self.v_proj = nn.Conv2d(self.feat_dim, self.embed_dim, 1).to(self.device)
        if hasattr(self.visual, 'proj') and self.visual.proj is not None:
            with torch.no_grad():
                self.v_proj.weight.data.copy_(self.visual.proj.data.T.unsqueeze(-1).unsqueeze(-1))

        self.classnames = classnames
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)
        self._init_zeroshot_classifier()

    @torch.no_grad()
    def _init_zeroshot_classifier(self):
        text_embeds = []
        for cls_name in self.classnames:
            tokens = open_clip.tokenize([f'a photo of a {cls_name}']).to(self.device)
            embed = F.normalize(self.clip_model.encode_text(tokens), dim=-1)
            text_embeds.append(embed.squeeze(0))
        self.zeroshot_weights = torch.stack(text_embeds, dim=1) 

    def _stem(self, x):
        B, _, H, W = x.shape
        x_in = self.visual.conv1(x) 
        gh, gw = x_in.shape[2], x_in.shape[3]
        
        pos_embed = self.visual.positional_embedding
        cls_pos, patch_pos = pos_embed[:1, :], pos_embed[1:, :]
        orig_size = int(patch_pos.shape[0]**0.5)
        patch_pos = F.interpolate(patch_pos.reshape(1, orig_size, orig_size, -1).permute(0, 3, 1, 2), 
                                  size=(gh, gw), mode='bicubic', align_corners=False)
        new_pos = torch.cat([cls_pos, patch_pos.permute(0, 2, 3, 1).reshape(gh*gw, -1)], dim=0)

        x_tokens = torch.cat([self.visual.class_embedding.expand(B, 1, -1), 
                             x_in.flatten(2).permute(0, 2, 1)], dim=1) + new_pos
        x_tokens = self.visual.ln_pre(x_tokens)

        resblocks = self.visual.transformer.resblocks
        for i in range(len(resblocks) - 1): x_tokens = resblocks[i](x_tokens)
        
        last_blk = resblocks[-1]
        x_norm = last_blk.ln_1(x_tokens)
        qkv = F.linear(x_norm, last_blk.attn.in_proj_weight, last_blk.attn.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        
        head_dim = q.shape[-1] // last_blk.attn.num_heads
        q = q.view(B, -1, last_blk.attn.num_heads, head_dim).transpose(1, 2)
        v = v.view(B, -1, last_blk.attn.num_heads, head_dim).transpose(1, 2)
        attn = (q @ q.transpose(-2, -1)) * (head_dim**-0.5)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, -1, q.shape[-1] * last_blk.attn.num_heads)
        
        feat = self.visual.ln_post(F.linear(out, last_blk.attn.out_proj.weight, last_blk.attn.out_proj.bias))
        return feat[:, 1:, :].permute(0, 2, 1).reshape(B, self.feat_dim, gh, gw)

    def forward(self, x):
        # 1. 提取 CLIP 原始特征
        lr_feat = self._stem(x) 
        feat_reduced = F.normalize(self.v_proj(lr_feat), dim=1)
        
        # 2. 生成低分辨率 Logits
        B, C, gh, gw = feat_reduced.shape
        logits = (feat_reduced.permute(0, 2, 3, 1).reshape(-1, C) @ self.zeroshot_weights) / self.temperature
        logits = logits.reshape(B, gh, gw, -1).permute(0, 3, 1, 2) 

        # 3. 🔥 核心修改：使用原图 x 作为引导
        # 注意：这里 x 是归一化后的，如果想让颜色引导更强，也可以传入未归一化的图
        refined_logits = self.pamr(x, logits)
        return refined_logits

# ==========================================
# 3. 推理逻辑
# ==========================================

def run_inference(image_path, labels, palette, output_path="clearclip_final.png"):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DenseClip("ViT-B-16", labels, device).eval()

    img_raw = Image.open(image_path).convert('RGB')
    orig_w, orig_h = img_raw.size
    
    # 稍微增大处理分辨率以获得更精细的 PAMR 结果
    target_w = 640 
    ratio = target_w / orig_w
    target_h = int(orig_h * ratio)
    new_w = (target_w // 16) * 16
    new_h = (target_h // 16) * 16
    
    transform = transforms.Compose([
        transforms.Resize((new_h, new_w)),
        transforms.ToTensor(),
        transforms.Normalize((0.481, 0.457, 0.408), (0.268, 0.261, 0.275))
    ])
    
    with torch.inference_mode():
        img_tensor = transform(img_raw).unsqueeze(0).to(device)
        # 此时 model 返回的结果分辨率已经是 (new_h, new_w)
        output = model(img_tensor)
        mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    # 映射颜色
    color_mask = palette[mask]
    final_img = Image.fromarray(color_mask.astype(np.uint8))
    
    # 最终缩放回原图尺寸
    final_img = final_img.resize((orig_w, orig_h), Image.BILINEAR)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_img.save(output_path)
    print(f"Result saved to {output_path}")

# ==========================================
# 4. 执行
# ==========================================

target_labels = [
    'background', 'bareland', 'pavement', 'road', 'water', 
    'tree', 'grass', 'cropland', 'building'
]

# 对应类别的调色盘 (Viridis 风格)
custom_palette = np.array([
    (68, 1, 84), (72, 40, 120), (62, 74, 137), (49, 104, 142), (38, 130, 142), 
    (31, 158, 137), (73, 193, 110), (160, 218, 57), (253, 231, 37)
], dtype=np.uint8)

if __name__ == "__main__":
    # 确保 img3.jpg 存在
    run_inference("./dataset/UDD/DJI_0591.JPG", target_labels, custom_palette, "./benchmark/UDD/clearclip/DJI_0591.JPG")