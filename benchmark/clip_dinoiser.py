import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import numpy as np
from open_clip import get_tokenizer, create_model_from_pretrained
from typing import List

# --- 1. 标准化配置 ---
# CLIP 使用 OpenAI 的归一化，DINO 通常使用 ImageNet 的归一化
CLIP_NORM = T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
DINO_NORM = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
IMAGENET_TEMPLATES = ['a photo of a {}.', 'a segmentation of a {}.', 'the {} in the scene.']

# --- 2. DINO 提取器 (基于你提供的代码进行封装) ---
class DINO_Extractor(nn.Module):
    def __init__(self):
        super().__init__()
        # 使用第一代 DINO ViT-B/16
        self.backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
        self.hook_features = {}
        self.patch_size = 16
        
        # 注册 QKV Hook
        def hook_fn_forward_qkv(module, input, output):
            self.hook_features["qkv"] = output
            
        self.backbone._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(hook_fn_forward_qkv)

    @torch.no_grad()
    def forward(self, x, type_feats="k"):
        # DINO 支路使用自己的归一化
        _ = self.backbone(DINO_NORM(x))
        
        # 提取 QKV
        nh = self.backbone.blocks[-1].attn.num_heads
        nb_im, nb_tokens, C_qkv = self.hook_features["qkv"].shape
        qkv = (self.hook_features["qkv"]
               .reshape(nb_im, nb_tokens, 3, nh, C_qkv // nh // 3)
               .permute(2, 0, 3, 1, 4))
        
        # q, k, v 形状: [B, nh, N, d]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        if type_feats == "k":
            feats = k.transpose(1, 2).flatten(-2, -1) # [B, N, C]
        elif type_feats == "v":
            feats = v.transpose(1, 2).flatten(-2, -1)
        else:
            feats = q.transpose(1, 2).flatten(-2, -1)
            
        # 移除 CLS token (index 0)
        feats = feats[:, 1:, :] 
        # 归一化
        feats = F.normalize(feats, dim=-1)
        return feats

# --- 3. MaskCLIP 语义支路 ---
class MaskCLIP_Backbone(nn.Module):
    def __init__(self, model_name="ViT-B-16", pretrained="laion2b_s34b_b88k"):
        super().__init__()
        model, _ = create_model_from_pretrained(model_name, pretrained=pretrained)
        self.clip = model.eval()
        self.tokenizer = get_tokenizer(model_name)
        self.patch_size = 16
        
        # 备份原始位置编码用于插值
        self.orig_pos_embed = nn.Parameter(self.clip.visual.positional_embedding.data.clone())
        
        # 修改投影层为卷积
        self.proj = nn.Conv2d(768, 512, 1, bias=False)
        self.proj.weight = nn.Parameter(self.clip.visual.proj.t()[:, :, None, None])
        
        # Hook 倒数第二层
        self.v_feat = {}
        def hook_v(m, i, o): self.v_feat["v"] = o
        self.clip.visual.transformer.resblocks[-2].register_forward_hook(hook_v)

    def _resize_pos(self, h, w):
        cls_pos = self.orig_pos_embed[0:1]
        spatial_pos = self.orig_pos_embed[1:]
        old_size = int(np.sqrt(spatial_pos.shape[0]))
        spatial_pos = spatial_pos.reshape(1, old_size, old_size, -1).permute(0, 3, 1, 2)
        spatial_pos = F.interpolate(spatial_pos, size=(h//self.patch_size, w//self.patch_size), mode='bicubic')
        spatial_pos = spatial_pos.permute(0, 2, 3, 1).reshape(-1, 768)
        return torch.cat([cls_pos, spatial_pos], dim=0)

    @torch.no_grad()
    def forward(self, x):
        B, _, H, W = x.shape
        # 动态插值位置编码
        self.clip.visual.positional_embedding = nn.Parameter(self._resize_pos(H, W))
        
        # 提取 CLIP 特征
        _ = self.clip(CLIP_NORM(x))
        v = self.v_feat["v"].permute(1, 0, 2) # [B, N, C]
        v = self.clip.visual.ln_post(v)
        
        # 转为特征图并投影
        v = v[:, 1:, :].permute(0, 2, 1) # [B, 768, N_patches]
        feat_map = v.reshape(B, 768, H//16, W//16)
        return F.normalize(self.proj(feat_map), dim=1)

# --- 4. 缝合与优化逻辑 ---
def refine_predictions(clip_sim, dino_k_feats, temp=0.04):
    """
    clip_sim: [K, N] (类别相似度)
    dino_k_feats: [N, C] (DINO 的 Key 特征)
    """
    # 计算 DINO 的自相似度矩阵 (Affinity)
    # A = K * K^T
    affinity = torch.matmul(dino_k_feats, dino_k_feats.t()) # [N, N]
    affinity = F.softmax(affinity / temp, dim=-1)
    
    # 语义传播: 将 CLIP 的预测在 DINO 的连通区域内平滑
    # refined = Clip_Sim * Affinity
    refined_sim = torch.matmul(clip_sim, affinity.t())
    return refined_sim

# --- 5. 执行主函数 ---
def run_stitched_inference(image_path, labels, output_path="final_stitched_seg.png"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 初始化两条支路
    clip_branch = MaskCLIP_Backbone().to(device).eval()
    dino_branch = DINO_Extractor().to(device).eval()
    
    # 读取图片
    raw_img = Image.open(image_path).convert('RGB')
    W_orig, H_orig = raw_img.size
    img_size = 448 # 必须是 16 的倍数
    transform = T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])
    img_tensor = transform(raw_img).unsqueeze(0).to(device)

    # 1. 获取文本分类器
    weights = []
    for label in labels:
        t = clip_branch.tokenizer([tmpl.format(label) for tmpl in IMAGENET_TEMPLATES]).to(device)
        emb = clip_branch.clip.encode_text(t)
        weights.append(F.normalize(emb, dim=-1).mean(dim=0))
    text_classifier = F.normalize(torch.stack(weights), dim=-1) # [K, 512]

    # 2. 提取特征
    clip_map = clip_branch(img_tensor) # [1, 512, 28, 28]
    dino_k = dino_branch(img_tensor, type_feats="k")[0] # [784, 768] (28*28=784)

    # 3. 计算原始 CLIP 相似度并展平
    K_classes = len(labels)
    clip_sim = torch.einsum('bchw,kc->khw', clip_map, text_classifier) # [K, 28, 28]
    clip_sim_flat = clip_sim.flatten(1) # [K, 784]

    # 4. 使用 DINO Key 进行缝合优化
    refined_sim_flat = refine_predictions(clip_sim_flat, dino_k)
    refined_sim = refined_sim_flat.reshape(K_classes, 28, 28)

    # 5. 上采样并生成 Mask
    refined_sim = F.interpolate(refined_sim.unsqueeze(0), size=(H_orig, W_orig), mode='bilinear').squeeze()
    mask = refined_sim.argmax(dim=0).cpu().numpy()

    # 6. 渲染 (遥感配色)
    palette = [
        (0,0,0), (139,69,19), (128,128,128), (50,50,50), (0,0,255),
        (0,100,0), (0,255,0), (255,255,0), (255,0,0)
    ]
    color_mask = np.zeros((H_orig, W_orig, 3), dtype=np.uint8)
    for idx, color in enumerate(palette[:len(labels)]):
        color_mask[mask == idx] = color

    Image.fromarray(color_mask).save(output_path)
    print(f"🚀 缝合完成！结果已保存至: {output_path}")

if __name__ == "__main__":
    target_labels = [
        'background', 'bareland', 'pavement', 'road', 'water',
        'tree', 'grass', 'cropland', 'building'
    ]
    run_stitched_inference("img2.jpg", target_labels)