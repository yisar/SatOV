import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from torch import Tensor
from open_clip import get_tokenizer, create_model_from_pretrained
import torchvision.transforms as T
from PIL import Image
import numpy as np
from functools import partial

# ============================
# 1. PAMR 优化模块（直接加入）
# ============================
class LocalAffinity(nn.Module):
    def __init__(self, dilations=[1]):
        super().__init__()
        self.dilations = dilations
        weight = self._init_aff()
        self.register_buffer('kernel', weight)

    def _init_aff(self):
        weight = torch.zeros(8, 1, 3, 3)
        for i in range(weight.size(0)):
            weight[i, 0, 1, 1] = 1
        weight[0, 0, 0, 0] = -1
        weight[1, 0, 0, 1] = -1
        weight[2, 0, 0, 2] = -1
        weight[3, 0, 1, 0] = -1
        weight[4, 0, 1, 2] = -1
        weight[5, 0, 2, 0] = -1
        weight[6, 0, 2, 1] = -1
        weight[7, 0, 2, 2] = -1
        self.weight_check = weight.clone()
        return weight

    def forward(self, x):
        self.weight_check = self.weight_check.type_as(x)
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
        weight[0, 0, 0, 0] = 1
        weight[1, 0, 0, 1] = 1
        weight[2, 0, 0, 2] = 1
        weight[3, 0, 1, 0] = 1
        weight[4, 0, 1, 2] = 1
        weight[5, 0, 2, 0] = 1
        weight[6, 0, 2, 1] = 1
        weight[7, 0, 2, 2] = 1
        self.weight_check = weight.clone()
        return weight

class LocalStDev(LocalAffinity):
    def _init_aff(self):
        weight = torch.zeros(9, 1, 3, 3)
        weight[0, 0, 0, 0] = 1
        weight[1, 0, 0, 1] = 1
        weight[2, 0, 0, 2] = 1
        weight[3, 0, 1, 0] = 1
        weight[4, 0, 1, 1] = 1
        weight[5, 0, 1, 2] = 1
        weight[6, 0, 2, 0] = 1
        weight[7, 0, 2, 1] = 1
        weight[8, 0, 2, 2] = 1
        self.weight_check = weight.clone()
        return weight

    def forward(self, x):
        x = super().forward(x)
        return x.std(2, keepdim=True)

class LocalAffinityAbs(LocalAffinity):
    def forward(self, x):
        x = super().forward(x)
        return torch.abs(x)

class PAMR(nn.Module):
    def __init__(self, num_iter=3, dilations=[1,2]):  # 迭代3次效果最好
        super().__init__()
        self.num_iter = num_iter
        self.aff_x = LocalAffinityAbs(dilations)
        self.aff_m = LocalAffinityCopy(dilations)
        self.aff_std = LocalStDev(dilations)

    def forward(self, x, mask):
        mask = F.interpolate(mask, size=x.size()[-2:], mode="bilinear", align_corners=True)
        B, K, H, W = x.size()
        _, C, _, _ = mask.size()
        x_std = self.aff_std(x)
        x = -self.aff_x(x) / (1e-8 + 0.1 * x_std)
        x = x.mean(1, keepdim=True)
        x = F.softmax(x, 2)

        for _ in range(self.num_iter):
            m = self.aff_m(mask)
            mask = (m * x).sum(2)
        return mask

# ============================
# 2. 你原来的 MaskClip 代码
# ============================
imagenet_templates = [
    'a photo of a {}.',
    'a bad photo of a {}.',
    'a segmentation of a {}.',
    'a photo of many {}.',
    'the {} in the scene.',
    'a close-up photo of a {}.',
]
OPENAI_NORMALIZE = T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

class MaskClip(nn.Module):
    def __init__(self, clip_model="ViT-B-16", pretrained="laion2b_s34b_b88k", patch_size=16, img_size=(224, 224), in_channels=768, text_channels=512):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        model, _ = create_model_from_pretrained(clip_model, pretrained=pretrained)
        model.eval()
        self.clip_T = OPENAI_NORMALIZE
        self.hook_features = {}
        self.backbone = model
        def hook_fn_forward(module, input, output):
            self.hook_features["v"] = output
        self.backbone.visual.transformer.resblocks[-2].register_forward_hook(hook_fn_forward)
        self._positional_embd = nn.Parameter(self.backbone.visual.positional_embedding.data.clone())
        self.proj = nn.Conv2d(in_channels, text_channels, 1, bias=False)
        self.proj.weight = nn.Parameter(model.visual.proj.t()[:, :, None, None])
        self.tokenizer = get_tokenizer(clip_model)

    @torch.no_grad()
    def extract_feat(self, inputs: Tensor) -> Tensor:
        pos_embed = self.backbone.visual.positional_embedding
        B, C, H, W = inputs.shape
        hw_shape = (H // self.patch_size, W // self.patch_size)
        x_len, pos_len = hw_shape[0] * hw_shape[1], pos_embed.shape[0]
        if x_len != pos_len - 1:
            pos_h = self.img_size[0] // self.patch_size
            pos_w = self.img_size[1] // self.patch_size
            self.backbone.visual.positional_embedding.data = self.resize_pos_embed(
                self._positional_embd[None], hw_shape, (pos_h, pos_w), 'bicubic')[0]
        _ = self.backbone(inputs)
        v = self.hook_features["v"]
        v = self.extract_v(v, self.backbone.visual.transformer.resblocks[-1]).permute(1, 0, 2)
        v = self.backbone.visual.ln_post(v)
        v = v.permute(1, 0, 2)[:, 1:]
        v = v.reshape(B, hw_shape[0], hw_shape[1], -1).permute(0, 3, 1, 2).contiguous()
        self.backbone.visual.positional_embedding.data = self._positional_embd
        return v

    @torch.no_grad()
    def extract_v(self, x, block):
        y = block.ln_1(x)
        y = F.linear(y, block.attn.in_proj_weight, block.attn.in_proj_bias)
        B, N, C = y.shape
        y = y.view(B, N, 3, C // 3).permute(2, 0, 1, 3).reshape(3 * B, N, C // 3)
        y = F.linear(y, block.attn.out_proj.weight, block.attn.out_proj.bias)
        q, k, v = y.tensor_split(3, dim=0)
        v += x
        v += block.mlp(block.ln_2(v))
        return v

    @staticmethod
    def resize_pos_embed(pos_embed, input_shpae, pos_shape, mode):
        pos_h, pos_w = pos_shape
        cls_token_weight = pos_embed[:, 0]
        pos_embed_weight = pos_embed[:, 1:]
        pos_embed_weight = pos_embed_weight.reshape(1, pos_h, pos_w, pos_embed.shape[2]).permute(0, 3, 1, 2)
        pos_embed_weight = F.interpolate(pos_embed_weight, size=input_shpae, align_corners=False, mode=mode)
        cls_token_weight = cls_token_weight.unsqueeze(1)
        pos_embed_weight = torch.flatten(pos_embed_weight, 2).transpose(1, 2)
        return torch.cat((cls_token_weight, pos_embed_weight), dim=1)

    @torch.no_grad()
    def get_classifier(self, classnames: List[str]) -> Tensor:
        device = next(self.parameters()).device
        aug_embeddings = torch.stack([self._embed_label(label, device) for label in classnames])
        return F.normalize(aug_embeddings, dim=-1)

    def _embed_label(self, label: str, device) -> Tensor:
        all_prompts = self.tokenizer([template.format(label) for template in imagenet_templates]).to(device)
        out = self.backbone.encode_text(all_prompts)
        out = F.normalize(out, dim=-1)
        return out.mean(dim=0)

    @torch.no_grad()
    def forward(self, inputs: Tensor) -> Tensor:
        inputs = self.clip_T(inputs)
        x = self.extract_feat(inputs)
        feats = self.proj(x)
        return feats

# ============================
# 3. 推理函数（已集成 PAMR）
# ============================
def run_inference(image_path, labels, save_path="segmentation_result.png"):
    custom_palette = [
        (68, 1, 84), (72, 40, 120), (62, 74, 137), (49, 104, 142),
        (38, 130, 142), (31, 158, 137), (73, 193, 110), (160, 218, 57), (253, 231, 37)
    ]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 模型
    model = MaskClip().to(device).eval()
    pamr = PAMR(num_iter=3, dilations=[1,2]).to(device).eval()  # PAMR初始化

    # 图像
    raw_img = Image.open(image_path).convert('RGB')
    input_size = (448, 448)
    transform = T.Compose([T.Resize(input_size), T.ToTensor()])
    img_tensor = transform(raw_img).unsqueeze(0).to(device)

    # 1. 提取特征
    img_feats = model(img_tensor)
    img_feats = F.normalize(img_feats, dim=1)
    text_feats = model.get_classifier(labels)

    # 2. 相似度图
    similarity = torch.einsum('bchw,kc->bkhw', img_feats, text_feats)
    similarity = F.interpolate(similarity, size=raw_img.size[::-1], mode='bilinear')

    # ======================
    # ✨ 核心：PAMR 优化
    # ======================
    with torch.no_grad():
        refined_similarity = pamr(img_tensor, similarity)  # 用原图引导优化分割图

    # 生成掩码
    mask = refined_similarity.argmax(1).squeeze().cpu().numpy()

    # 上色
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, color in enumerate(custom_palette):
        color_mask[mask == idx] = color

    seg_img = Image.fromarray(color_mask)
    seg_img.save(save_path)
    print(f"✅ 优化完成！结果已保存至：{save_path}")

# ============================
# 执行
# ============================
if __name__ == "__main__":
    target_labels = [
        'background', 'bareland', 'pavement', 'road','water',
        'tree', 'grass', 'cropland', 'building'
    ]
    run_inference("img2.jpg", target_labels, save_path="maskclip_pamr.png")