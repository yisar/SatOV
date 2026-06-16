import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from PIL import Image
import numpy as np


# --------------------------
# JAFAR 上采样模块（论文标准实现）
# --------------------------
class JAFARUpSampler(nn.Module):
    def __init__(
        self,
        feat_dim: int = 768,
        hidden_dim: int = 512,
        up_scale: int = 4,
        num_heads: int = 8,
    ):
        super().__init__()
        self.scale = up_scale
        self.feat_dim = feat_dim

        # 低层纹理引导分支
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GELU(),
        )
        self.low_proj = nn.Conv2d(128, hidden_dim, 1)

        # 交叉注意力
        self.q_proj = nn.Linear(feat_dim, hidden_dim)
        self.kv_proj = nn.Linear(hidden_dim, hidden_dim * 2)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

        # 4倍上采样重建
        self.up_blocks = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(hidden_dim // 2, feat_dim, 4, stride=2, padding=1),
        )
        self.norm = nn.LayerNorm(feat_dim)

    def forward(self, clip_feat: torch.Tensor, raw_img: torch.Tensor):
        """
        clip_feat: [B, 1+N_patch, C] 包含class token
        raw_img: 归一化原图 [B,3,H,W]
        return: 高分辨率稠密特征 [B, C, H*4, W*4]
        """
        B, N, C = clip_feat.shape
        Hp = Wp = int(np.sqrt(N - 1))
        feat_map = clip_feat[:, 1:, :].reshape(B, Hp, Wp, C).permute(0, 3, 1, 2)

        # 提取低层纹理
        low_feat = self.low_level_conv(raw_img)
        low_feat = self.low_proj(low_feat)
        B_l, C_l, H_l, W_l = low_feat.shape
        low_flat = low_feat.permute(0, 2, 3, 1).reshape(B_l, -1, C_l)

        # Cross Attention
        q = self.q_proj(clip_feat)
        k, v = torch.chunk(self.kv_proj(low_flat), 2, dim=-1)
        attn_out, _ = self.attn(q, k, v)

        attn_map = attn_out[:, 1:, :].reshape(B, Hp, Wp, -1).permute(0, 3, 1, 2)
        high_feat = self.up_blocks(attn_map)

        high_feat = self.norm(high_feat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return high_feat


# --------------------------
# OpenAI CLIP + JAFAR 推理封装
# --------------------------
class OpenAICLIPJAFARRS(nn.Module):
    def __init__(
        self,
        clip_model_name: str = "ViT-B/16",
        jafar_ckpt_path: str = "./jafar.pth",
        device=None,
    ):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 加载原生 OpenAI CLIP
        self.clip_model, self.clip_preprocess = clip.load(
            clip_model_name, device=self.device
        )
        self.clip_model.eval()
        for p in self.clip_model.parameters():
            p.requires_grad = False

        # 初始化JAFAR并加载预训练权重
        self.jafar = JAFARUpSampler(feat_dim=768, up_scale=4).to(self.device)
        ckpt = torch.load(jafar_ckpt_path, map_location=self.device)
        self.jafar.load_state_dict(ckpt["jafar_state_dict"], strict=True)
        self.jafar.eval()

    @torch.no_grad()
    def forward(self, pil_img: Image.Image):
        """
        输入RGB遥感PIL图，输出4倍上采样稠密CLIP特征
        out: [1, 768, H*4, W*4]
        """
        # CLIP预处理
        img_tensor = self.clip_preprocess(pil_img).unsqueeze(0).to(self.device)
        # 提取ViT patch特征（带class token）
        clip_feat = self.clip_model.encode_image_full(img_tensor)
        # JAFAR上采样
        dense_hr_feat = self.jafar(clip_feat, img_tensor)
        return dense_hr_feat


# --------------------------
# Demo 运行入口
# --------------------------
if __name__ == "__main__":
    # 1. 初始化
    jafar_weight_file = "./jafar.pth"
    model = OpenAICLIPJAFARRS(
        clip_model_name="ViT-B/16", jafar_ckpt_path=jafar_weight_file
    )

    # 2. 读取RGB遥感影像
    rs_image = Image.open("remote_sensing_rgb.png").convert("RGB")
    print(f"输入遥感图尺寸 W×H: {rs_image.size}")

    # 3. 推理高分辨率稠密特征
    hr_feature = model(rs_image)
    print(f"JAFAR输出稠密特征 shape: {hr_feature.shape}")
    # [1, 768, H*4, W*4]

    # 4. 转numpy供下游分割/解译使用
    feat_np = hr_feature.squeeze(0).cpu().numpy()
    print("稠密CLIP特征提取完成，可接入分割头/开放词汇解译模块")
