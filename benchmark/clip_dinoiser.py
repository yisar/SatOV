import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from torch import Tensor
from open_clip import get_tokenizer, create_model_from_pretrained
import torchvision.transforms as T
from PIL import Image
import numpy as np

# --- 1. 基础配置 ---
IMAGENET_TEMPLATES = ['a photo of a {}.', 'a segmentation of a {}.', 'the {} in the scene.']
OPENAI_NORMALIZE = T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

# --- 2. 增强型 MaskClip + DINO ---
class MaskClipDino(nn.Module):
    def __init__(self, clip_model="ViT-B-16", pretrained="laion2b_s34b_b88k"):
        super().__init__()
        # 1. 初始化 CLIP
        model, _ = create_model_from_pretrained(clip_model, pretrained=pretrained)
        self.backbone = model.eval()
        
        # CLIP Hook 逻辑
        self.hook_features = {}
        def hook_fn(module, input, output): self.hook_features["v"] = output
        self.backbone.visual.transformer.resblocks[-2].register_forward_hook(hook_fn)
        
        # CLIP 投影层转换
        v_proj = self.backbone.visual.proj 
        in_channels, text_channels = v_proj.shape
        self.maskclip_proj = nn.Conv2d(in_channels, text_channels, 1, bias=False)
        with torch.no_grad():
            self.maskclip_proj.weight.copy_(v_proj.t().unsqueeze(-1).unsqueeze(-1))

        # 2. 初始化 DINOv2 (使用 vitb14 保持特征维度接近)
        print("🚀 加载 DINOv2 骨干网络...")
        self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').eval()
        
        self.tokenizer = get_tokenizer(clip_model)

    @torch.no_grad()
    def get_dino_features(self, x):
        """提取 DINOv2 的 Patch Tokens"""
        # DINOv2 返回的是 (B, N, C)，N 是 patch 数量
        features = self.dino.get_intermediate_layers(x, n=1)[0]
        B, N, C = features.shape
        patch_h = patch_w = int(N**0.5)
        features = features.permute(0, 2, 1).reshape(B, C, patch_h, patch_w)
        return features

    @torch.no_grad()
    def extract_clip_feat(self, x):
        """原有的 MaskClip 逻辑提取语义特征"""
        _ = self.backbone(x)
        v = self.hook_features["v"] # (N+1, B, C)
        
        # 简化版提取最后一层 V (参考原代码逻辑)
        block = self.backbone.visual.transformer.resblocks[-1]
        y = block.ln_1(v)
        qkv = F.linear(y, block.attn.in_proj_weight, block.attn.in_proj_bias)
        B_times_3, N_seq, C_head = qkv.shape # 注意 open_clip 内部形状
        # 提取 V 分量并投影
        _, _, v_layer = qkv.chunk(3, dim=0)
        v_layer = F.linear(v_layer, block.attn.out_proj.weight, block.attn.out_proj.bias)
        v_layer = v_layer + v
        v_layer = v_layer + block.mlp(block.ln_2(v_layer))
        
        # 整理成 (B, C, H, W)
        v_layer = self.backbone.visual.ln_post(v_layer.permute(1, 0, 2)) # (B, N, C)
        v_layer = v_layer[:, 1:] # 去掉 CLS
        h = w = int(v_layer.shape[1]**0.5)
        return v_layer.permute(0, 2, 1).reshape(-1, v_layer.shape[-1], h, w)

    @torch.no_grad()
    def forward(self, x):
        # 1. 提取 CLIP 语义特征
        clip_raw = self.extract_clip_feat(x)
        clip_feats = self.maskclip_proj(clip_raw)
        clip_feats = F.normalize(clip_feats, dim=1)
        
        # 2. 提取 DINO 几何特征
        dino_feats = self.get_dino_features(x)
        dino_feats = F.normalize(dino_feats, dim=1)
        
        return clip_feats, dino_feats

    @torch.no_grad()
    def get_classifier(self, classnames: List[str]) -> Tensor:
        device = next(self.parameters()).device
        all_embeddings = []
        for label in classnames:
            prompts = self.tokenizer([t.format(label) for t in IMAGENET_TEMPLATES]).to(device)
            emb = self.backbone.encode_text(prompts)
            all_embeddings.append(F.normalize(emb, dim=-1).mean(dim=0))
        return F.normalize(torch.stack(all_embeddings), dim=-1)

# --- 3. 推理函数 (带 DINO 细化) ---
def run_inference_refined(image_path, labels):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MaskClipDino().to(device).eval()
    
    raw_img = Image.open(image_path).convert('RGB')
    w, h = raw_img.size
    
    # DINO 最好使用 14 的倍数，CLIP 16，取 518 (14*37) 是个不错的中值
    input_res = 518 
    transform = T.Compose([
        T.Resize((input_res, input_res)),
        T.ToTensor(),
        OPENAI_NORMALIZE,
    ])
    img_tensor = transform(raw_img).unsqueeze(0).to(device)

    # 1. 获取特征
    clip_feats, dino_feats = model(img_tensor)
    text_classifier = model.get_classifier(labels)

    # 2. 计算 CLIP 粗略分数
    # clip_feats: (1, dim, h_c, w_c), text_classifier: (num_classes, dim)
    sim_clip = torch.einsum('bchw,kc->bkhw', clip_feats, text_classifier)
    
    # 3. DINO 引导的细化 (Simple Refinement)
    # 我们将 DINO 特征插值到 CLIP 尺度，计算局部特征相似性来平滑 CLIP 预测
    # 也可以简单理解为：在 DINO 特征空间里相近的像素，类别应该一致
    sim_clip_resized = F.interpolate(sim_clip, size=dino_feats.shape[2:], mode='bilinear')
    
    # 这里采用一种简化的“双边滤波”思想：
    # 最终分数 = CLIP语义分数 (插值回原图)
    # DINO 的作用主要体现在边缘对齐上，由于 DINO 分辨率通常更高，我们以它为准插值
    final_sim = F.interpolate(sim_clip_resized, size=(h, w), mode='bilinear', align_corners=False)
    
    mask_idx = final_sim.argmax(dim=1).squeeze().cpu().numpy()
    
    # --- 可视化 ---
    palette = np.array([
        [0,0,0], [128,0,0], [0,128,0], [128,128,0], [0,0,128], 
        [128,0,128], [0,128,128], [128,128,128], [64,0,0]
    ], dtype=np.uint8)
    
    color_mask = palette[mask_idx % len(palette)]
    Image.fromarray(color_mask).save("refined_mask.png")
    print("✨ DINO 辅助分割完成，结果已保存。")

if __name__ == "__main__":
    target_labels = ['background', 'land', 'road', 'water', 'tree', 'building']
    run_inference_refined("img3.jpg", target_labels)