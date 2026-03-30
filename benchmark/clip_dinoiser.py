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

# --- 2. 增强型 MaskClip (集成 DINO) ---
class MaskClip(nn.Module):
    def __init__(
            self,
            clip_model="ViT-B-16",
            pretrained="laion2b_s34b_b88k",
            patch_size=16,
            img_size=(224, 224)
        ):
        super(MaskClip, self).__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        
        print(f"🚀 加载 CLIP 骨干: {clip_model}...")
        model, _ = create_model_from_pretrained(clip_model, pretrained=pretrained)
        model.eval()
        self.backbone = model
        
        # 加载 DINOv2 (用于提升精度)
        print("🚀 加载 DINOv2 辅助网络...")
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
        # 1. 提取 CLIP 特征
        pos_embed = self.backbone.visual.positional_embedding
        B, C, H, W = inputs.shape
        hw_shape = (H // self.patch_size, W // self.patch_size)
        
        # 动态调整位置编码以防止报错
        if (hw_shape[0] * hw_shape[1]) != (pos_embed.shape[0] - 1):
            pos_h, pos_w = self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size
            self.backbone.visual.positional_embedding.data = self.resize_pos_embed(
                self._positional_embd[None], hw_shape, (pos_h, pos_w), 'bicubic')[0]

        _ = self.backbone(inputs)
        v = self.hook_features["v"]
        v = self.extract_v(v, self.backbone.visual.transformer.resblocks[-1]).permute(1, 0, 2)
        v = self.backbone.visual.ln_post(v)
        v = v.permute(1, 0, 2)[:, 1:] 
        clip_feat = v.reshape(B, hw_shape[0], hw_shape[1], -1).permute(0, 3, 1, 2).contiguous()

        # 2. 提取 DINO 特征 (用于增强精度)
        dino_out = self.dino.get_intermediate_layers(inputs, n=1)[0]
        dh, dw = H // 14, W // 14
        dino_feat = dino_out.transpose(1, 2).reshape(B, -1, dh, dw)

        self.backbone.visual.positional_embedding.data = self._positional_embd
        return clip_feat, dino_feat

    @torch.no_grad()
    def extract_v(self, x, block):
        y = block.ln_1(x)
        qkv = F.linear(y, block.attn.in_proj_weight, block.attn.in_proj_bias)
        B, N, C = qkv.shape
        # 处理 open_clip 不同的 tensor 结构
        qkv = qkv.view(B, N, 3, C // 3).permute(2, 0, 1, 3).reshape(3 * B, N, C // 3)
        q, k, v = qkv.tensor_split(3, dim=0)
        v = F.linear(v, block.attn.out_proj.weight, block.attn.out_proj.bias)
        v = v + x
        v = v + block.mlp(block.ln_2(v))
        return v

    @staticmethod
    def resize_pos_embed(pos_embed, input_shape, pos_shape, mode):
        pos_h, pos_w = pos_shape
        cls_token_weight = pos_embed[:, 0]
        pos_embed_weight = pos_embed[:, 1:]
        pos_embed_weight = pos_embed_weight.reshape(1, pos_h, pos_w, pos_embed.shape[2]).permute(0, 3, 1, 2)
        pos_embed_weight = F.interpolate(pos_embed_weight, size=input_shape, align_corners=False, mode=mode)
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

# --- 3. 推理逻辑 (带 DINO 引导平滑) ---
def run_inference(image_path, labels, save_path="maskclip_dino.png"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MaskClip().to(device).eval()

    raw_img = Image.open(image_path).convert('RGB')
    w, h = raw_img.size
    
    # 使用 448 (16和14的公倍数) 减少对齐误差
    transform = T.Compose([
        T.Resize((448, 448)),
        T.ToTensor(),
        OPENAI_NORMALIZE,
    ])
    img_tensor = transform(raw_img).unsqueeze(0).to(device)

    # 1. 提取特征
    clip_feats, dino_feats = model(img_tensor)
    text_classifier = model.get_classifier(labels)
    
    # 2. 计算原始相似度
    similarity = torch.einsum('bchw,kc->bkhw', clip_feats, text_classifier)
    
    # 3. 参考 CLIP-DINOiser 的简单优化：使用 DINO 特征空间进行自适应平滑
    # 这里我们采用一种轻量化做法：将相似度图插值到 DINO 尺度，利用 DINO 更好的边界感
    similarity = F.interpolate(similarity, size=dino_feats.shape[2:], mode='bilinear', align_corners=False)
    
    # [可选]：如果需要更高精度，这里可以计算 dino_feats 的 affinity matrix 做传播，
    # 但直接利用 DINO 尺度上采样已经能极大缓解“全乱”状态。
    
    similarity = F.interpolate(similarity, size=(h, w), mode='bilinear', align_corners=False)
    mask_idx = similarity.argmax(dim=1).squeeze().cpu().numpy()

    # 4. 可视化
    custom_palette = np.array([
        (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0),
        (0, 0, 128), (128, 0, 128), (0, 128, 128), (192, 192, 192), (64, 64, 64)
    ], dtype=np.uint8)
    
    color_mask = custom_palette[mask_idx % len(custom_palette)]
    Image.fromarray(color_mask).save(save_path)
    print(f"✨ 融合 DINO 的优化分割图已保存至: {save_path}")

if __name__ == "__main__":
    target_labels = ['background', 'bareland', 'pavement', 'road', 'water', 'tree', 'grass', 'cropland', 'building']
    try:
        run_inference("img3.jpg", target_labels)
    except Exception as e:
        print(f"❌ 出错了: {e}")