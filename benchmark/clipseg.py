import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import torchvision.transforms as T
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

# --- 1. 配置与初始化 ---
def get_clipseg_model(model_id="CIDAS/clipseg-rd64-refined"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = CLIPSegProcessor.from_pretrained(model_id)
    model = CLIPSegForImageSegmentation.from_pretrained(model_id)
    model.to(device).eval()
    return model, processor, device

# --- 2. 推理逻辑 ---
@torch.no_grad()
def run_inference(image_path, labels, save_path="clipseg_result.png"):
    # 自定义颜色映射 (对应 9 个类别)
    custom_palette = [
        (68, 1, 84), (72, 40, 120), (62, 74, 137), (49, 104, 142),
        (38, 130, 142), (31, 158, 137), (73, 193, 110), (160, 218, 57), (253, 231, 37)
    ]
    
    model, processor, device = get_clipseg_model()
    print(f"Using device: {device}")

    # 读取并预处理图像
    raw_img = Image.open(image_path).convert('RGB')
    width, height = raw_img.size

    # 1. 预处理：CLIPSeg 默认处理多条文本提示
    # 提示词增强（类似于 MaskClip 的模板）
    prompts = [f"a photo of {label}" for label in labels]
    
    inputs = processor(
        text=prompts, 
        images=[raw_img] * len(prompts), 
        padding="max_length", 
        return_tensors="pt"
    ).to(device)

    # 2. 前向传播
    # outputs.logits 形状为 [num_labels, 352, 352]
    outputs = model(**inputs)
    logits = outputs.logits
    
    # 3. 后处理：调整大小回原始尺寸
    # 如果只有 1 个标签，logits 维度是 [352, 352]，需要 unsqueeze
    if len(labels) == 1:
        logits = logits.unsqueeze(0)
    
    # 调整到原图大小
    # 注意：CLIPSeg 输出是 352x352 的热力图
    full_res_logits = F.interpolate(
        logits.unsqueeze(1), 
        size=(height, width), 
        mode='bilinear', 
        align_corners=False
    ).squeeze(1)

    # 4. 生成多类别掩码 (Argmax)
    # 每一个通道代表一个标签的概率，取最大值所在的索引作为类别
    preds = torch.sigmoid(full_res_logits)
    mask = torch.argmax(preds, dim=0).cpu().numpy()

    # 5. 渲染彩色图
    color_mask = np.zeros((height, width, 3), dtype=np.uint8)
    for idx, color in enumerate(custom_palette):
        if idx < len(labels):
            color_mask[mask == idx] = color

    # 6. 保存
    seg_img = Image.fromarray(color_mask)
    seg_img.save(save_path)
    print(f"✅ CLIPSeg 分割结果已保存至：{save_path}")

# --- 执行入口 ---
if __name__ == "__main__":
    # 需要安装：pip install transformers
    target_labels = [
        'background', 'bareland', 'pavement', 'road','water',
        'tree', 'grass', 'cropland', 'building'
    ]
    
    # 请确保 img2.jpg 在当前目录下
    try:
        run_inference("img2.jpg", target_labels, save_path="img2/clipseg_final.png")
    except Exception as e:
        print(f"错误: {e}. 请确保已安装 transformers 库并能访问 Hugging Face 权重。")