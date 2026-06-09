import math
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from transformers import CLIPModel, CLIPProcessor

# --- 1. 配置与初始化 (替换为标准 CLIP 模型) ---
def get_cat_clip_model(model_id="openai/clip-vit-base-patch16"):
    """
    使用原始 CLIP 模型提取密集特征。
    推荐使用 patch16 以获得更高的 Patch 分辨率 (如 14x14 的特征图)。
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 使用通用的 CLIPProcessor 和 CLIPModel
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id)
    model.to(device).eval()
    return model, processor, device

# --- 2. 推理逻辑 ---
@torch.no_grad()
def run_inference(image_path, labels, save_path="catclip_result.png"):
    # 自定义颜色映射 (对应 9 个类别)
    custom_palette = [
        (68, 1, 84), (72, 40, 120), (62, 74, 137), (49, 104, 142),
        (38, 130, 142), (31, 158, 137), (73, 193, 110), (160, 218, 57), (253, 231, 37)
    ]
    
    model, processor, device = get_cat_clip_model()
    print(f"Using device: {device}")

    # 读取图像
    raw_img = Image.open(image_path).convert('RGB')
    width, height = raw_img.size

    # 1. 预处理文本提示
    prompts = [f"a photo of {label}" for label in labels]
    text_inputs = processor(text=prompts, padding=True, return_tensors="pt").to(device)

    # 2. 预处理图像
    image_inputs = processor(images=raw_img, return_tensors="pt").to(device)

    # --- 核心：Cat-CLIP 密集特征匹配逻辑 ---
    
    # 3a. 获取文本特征并归一化
    text_features = model.get_text_features(**text_inputs)
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

    # 3b. 获取图像的 Patch 级特征
    # 首先通过 Vision Transformer 获取隐藏层状态
    vision_outputs = model.vision_model(pixel_values=image_inputs.pixel_values)
    
    # vision_outputs.last_hidden_state 形状: [1, seq_len, hidden_size]
    # seq_len = 1 (CLS token) + num_patches。我们需要去掉 CLS token，只保留 Patch 特征
    patch_embeds = vision_outputs.last_hidden_state[:, 1:, :]
    
    # 必须通过 visual_projection 将特征投影到与文本对齐的多模态空间
    patch_features = model.visual_projection(patch_embeds)
    patch_features = patch_features / patch_features.norm(p=2, dim=-1, keepdim=True) # [1, num_patches, embed_dim]

    # 4. 计算 Patch 和 文本 的余弦相似度 (Cosine Similarity)
    # patch_features: [1, N, D], text_features: [C, D] => logits: [1, N, C]
    logits = torch.matmul(patch_features, text_features.t())
    
    # 5. 重塑为 2D 空间网格
    # 对于 ViT，N = grid_size * grid_size (例如输入 224x224 且 patch=16 时，N=196，grid=14)
    num_patches = patch_features.shape[1]
    grid_size = int(math.sqrt(num_patches))
    
    # 维度转换: [1, N, C] -> [1, C, N] -> [1, C, grid_size, grid_size]
    logits = logits.permute(0, 2, 1).view(1, len(prompts), grid_size, grid_size)

    # 6. 后处理：插值回原图大小
    # 使用双线性插值将低分辨率的网格 (如 14x14) 上采样到原始图像分辨率
    full_res_logits = F.interpolate(
        logits, 
        size=(height, width), 
        mode='bilinear', 
        align_corners=False
    ).squeeze(0) # 形状变为 [num_labels, height, width]

    # 7. 生成多类别掩码 (Argmax)
    # 对于标准 CLIP，相似度本身可以作为概率分布，也可以用 softmax。这里我们直接取 argmax
    mask = torch.argmax(full_res_logits, dim=0).cpu().numpy()

    # 8. 渲染彩色图
    color_mask = np.zeros((height, width, 3), dtype=np.uint8)
    for idx, color in enumerate(custom_palette):
        if idx < len(labels):
            color_mask[mask == idx] = color

    # 9. 保存
    seg_img = Image.fromarray(color_mask)
    seg_img.save(save_path)
    print(f"✅ Cat-CLIP 分割结果已保存至：{save_path}")

# --- 执行入口 ---
if __name__ == "__main__":
    target_labels = [
        'background', 'bareland', 'pavement', 'road','water',
        'tree', 'grass', 'cropland', 'building'
    ]
    
    try:
        run_inference("./dataset/UDD/DJI_0591.JPG", target_labels, save_path="./benchmark/UDD/catclip/DJI_0591.JPG")
    except Exception as e:
        print(f"错误: {e}")