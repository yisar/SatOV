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
IMAGENET_TEMPLATES = [
    'a photo of a {}.',
    'a segmentation of a {}.',
    'the {} in the scene.',
    'a close-up photo of a {}.',
]

OPENAI_NORMALIZE = T.Normalize(
    (0.48145466, 0.4578275, 0.40821073),
    (0.26862954, 0.26130258, 0.27577711)
)

# --- 2. 增强型 MaskClip (集成 DINO 空间代理) ---
class MaskClip(nn.Module):
    def __init__(
            self,
            clip_model="ViT-B-16",
            pretrained="laion2b_s34b_b88k",
            patch_size=16,
            img_size=(448, 448)
        ):
        super(MaskClip, self).__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        
        print(f"🚀 加载 CLIP 骨干: {clip_model}...")
        model, _ = create_model_from_pretrained(clip_model, pretrained=pretrained)
        model.eval()
        self.backbone = model
        
        print("🚀 加载 DINOv2 辅助网络 (vitb14)...")
        self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').eval()
        
        self.hook_features = {}
        def hook_fn_forward(module, input, output):
            self.hook_features["v"] = output
        self.backbone.visual.transformer.resblocks[-2].register_forward_hook(hook_fn_forward)
        
        self._positional_embd = nn.Parameter(self.backbone.visual.positional_embedding.data.clone())
        
        v_proj = self.backbone.visual.proj 
        in_channels, text_channels = v_proj.shape
        self.maskclip_proj = nn.Conv2d(in_channels, text_channels, 1, bias=False)
        with torch.no_grad():
            self.maskclip_proj.weight.copy_(v_proj.t().unsqueeze(-1).unsqueeze(-1))
        
        self.tokenizer = get_tokenizer(clip_model)

    @torch.no_grad()
    def extract_feat(self, inputs: Tensor):
        B, C, H, W = inputs.shape
        hw_shape = (H // self.patch_size, W // self.patch_size)
        
        pos_embed = self.backbone.visual.positional_embedding
        if (hw_shape[0] * hw_shape[1]) != (self._positional_embd.shape[0] - 1):
            orig_num_patches = self._positional_embd.shape[0] - 1
            orig_h = int(orig_num_patches ** 0.5)
            orig_w = orig_num_patches // orig_h
            self.backbone.visual.positional_embedding.data = self.resize_pos_embed(
                self._positional_embd[None], target_shape=hw_shape, orig_shape=(orig_h, orig_w), mode='bicubic'
            )[0]

        _ = self.backbone(inputs)
        v = self.hook_features["v"]
        v = self.extract_v(v, self.backbone.visual.transformer.resblocks[-1]).permute(1, 0, 2)
        v = self.backbone.visual.ln_post(v)
        v = v.permute(1, 0, 2)[:, 1:] 
        clip_feat = v.reshape(B, hw_shape[0], hw_shape[1], -1).permute(0, 3, 1, 2).contiguous()

        dino_out = self.dino.get_intermediate_layers(inputs, n=1)[0]
        dh, dw = H // 14, W // 14
        dino_feat = dino_out.reshape(B, dh, dw, -1).permute(0, 3, 1, 2).contiguous()

        self.backbone.visual.positional_embedding.data = self._positional_embd
        return clip_feat, dino_feat

    @torch.no_grad()
    def extract_v(self, x, block):
        y = block.ln_1(x)
        qkv = F.linear(y, block.attn.in_proj_weight, block.attn.in_proj_bias)
        B, N, C = qkv.shape
        qkv = qkv.view(B, N, 3, C // 3).permute(2, 0, 1, 3).reshape(3 * B, N, C // 3)
        q, k, v = qkv.tensor_split(3, dim=0)
        v = F.linear(v, block.attn.out_proj.weight, block.attn.out_proj.bias)
        v = v + x
        v = v + block.mlp(block.ln_2(v))
        return v

    @staticmethod
    def resize_pos_embed(pos_embed, target_shape, orig_shape, mode):
        orig_h, orig_w = orig_shape
        cls_token_weight = pos_embed[:, 0]
        pos_embed_weight = pos_embed[:, 1:]
        pos_embed_weight = pos_embed_weight.reshape(1, orig_h, orig_w, pos_embed.shape[2]).permute(0, 3, 1, 2)
        pos_embed_weight = F.interpolate(pos_embed_weight, size=target_shape, align_corners=False, mode=mode)
        cls_token_weight = cls_token_weight.unsqueeze(1)
        pos_embed_weight = torch.flatten(pos_embed_weight, 2).transpose(1, 2)
        return torch.cat((cls_token_weight, pos_embed_weight), dim=1)

    @torch.no_grad()
    def get_classifier(self, classnames: List[str]) -> Tensor:
        device = next(self.parameters()).device
        aug_embeddings = torch.stack([self._embed_label(label, device) for label in classnames])
        return F.normalize(aug_embeddings, dim=-1)

    def _embed_label(self, label: str, device) -> Tensor:
        all_prompts = self.tokenizer([template.format(label) for template in IMAGENET_TEMPLATES]).to(device)
        out = self.backbone.encode_text(all_prompts)
        out = F.normalize(out, dim=-1)
        return out.mean(dim=0)

    @torch.no_grad()
    def forward(self, inputs: Tensor):
        clip_raw, dino_feat = self.extract_feat(inputs)
        clip_proj = self.maskclip_proj(clip_raw)
        clip_proj = F.normalize(clip_proj, dim=1)
        return clip_proj, dino_feat

# --- 3. 推理逻辑 (Proxy 机制核心修改区) ---
def run_inference(image_path, labels, save_path="./dataset/proxyclip_dino.png"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MaskClip().to(device).eval()

    raw_img = Image.open(image_path).convert('RGB')
    orig_w, orig_h = raw_img.size
    
    input_res = 448
    transform = T.Compose([
        T.Resize((input_res, input_res)),
        T.ToTensor(),
        OPENAI_NORMALIZE,
    ])
    img_tensor = transform(raw_img).unsqueeze(0).to(device)

    # 1. 提取基础特征
    clip_feats, dino_feats = model(img_tensor) # clip: [1, 512, 28, 28], dino: [1, 768, 32, 32]
    text_classifier = model.get_classifier(labels)
    
    # 2. 构建 DINO 空间代理 (Spatial Proxy)
    # 将 CLIP 特征对齐到 DINO 的空间分辨率 (32x32)
    B, C_clip, Hc, Wc = clip_feats.shape
    _, C_dino, Hd, Wd = dino_feats.shape
    clip_feats_resized = F.interpolate(clip_feats, size=(Hd, Wd), mode='bilinear', align_corners=False)
    
    # 计算 DINO 自亲和矩阵作为“空间代理权重”
    dino_norm = F.normalize(dino_feats, dim=1)
    dino_flat = dino_norm.view(B, C_dino, -1) # [B, 768, 1024]
    # Affinity: [B, 1024, 1024] -> 代表了像素间的空间结构关联
    spatial_proxy_affinity = torch.bmm(dino_flat.transpose(1, 2), dino_flat)
    spatial_proxy_affinity = F.softmax(spatial_proxy_affinity / 0.1, dim=-1)
    
    # 3. Proxy 引导的特征重构 (Feature Refinement)
    # 利用 DINO 的空间结构代理，重新分布 CLIP 的语义特征
    clip_flat = clip_feats_resized.view(B, C_clip, -1) # [B, 512, 1024]
    # 重构后的特征：每一个位置的语义都是由其在 DINO 空间下的“邻居”加权而来的
    refined_clip_flat = torch.bmm(clip_flat, spatial_proxy_affinity.transpose(1, 2))
    refined_clip_feats = refined_clip_flat.view(B, C_clip, Hd, Wd)
    
    # 4. 计算最终语义相似度
    # 使用重构后的、具有更好空间一致性的特征进行分类
    similarity = torch.einsum('bchw,kc->bkhw', refined_clip_feats, text_classifier)

    # 5. 上采样到原图大小并提取类别
    similarity = F.interpolate(similarity, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
    mask_idx = similarity.argmax(dim=1).squeeze().cpu().numpy()

    # 6. 可视化
    custom_palette = np.array([
        (68, 1, 84), (72, 40, 120), (62, 74, 137), (49, 104, 142),
        (38, 130, 142), (31, 158, 137), (73, 193, 110), (160, 218, 57), (253, 231, 37)
    ], dtype=np.uint8)
    
    color_mask = custom_palette[mask_idx % len(custom_palette)]
    Image.fromarray(color_mask).save(save_path)
    print(f"✨ 已通过 DINO 空间代理重构特征，结果已保存: {save_path}")

if __name__ == "__main__":
    target_labels = ['background', 'bareland', 'pavement', 'road', 'water', 'tree', 'grass', 'cropland', 'building']
    try:
        run_inference("./dataset/SSSI/6930.jpg", target_labels,"./benchmark/SSSI/proxyclip/6930.jpg")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ 出错了: {e}")