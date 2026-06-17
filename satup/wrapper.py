import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import open_clip
import torchvision.transforms as T


class CLIPWrapper(nn.Module):
    """
    RADIO-style wrapper for OpenCLIP ViT-B/16
    Output: spatial feature map (NCHW)
    """

    def __init__(self, name="ViT-B-16", pretrained="openai", device="cuda"):
        super().__init__()

        self.device = device

        # load CLIP
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            name, pretrained=pretrained
        )

        self.visual = self.model.visual.to(device).eval()

        for p in self.visual.parameters():
            p.requires_grad = False

        self.embed_dim = 768
        self.patch_size = 16

    def preprocess_img(self, img: Image.Image):
        """
        RADIO-style preprocessing hook
        """
        img = self.preprocess(img)
        return img.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def forward(self, img: Image.Image):
        """
        Returns:
            spatial_features: (1, 768, 14, 14)
            cls_token: None (for compatibility with RADIO)
        """

        x = self.preprocess_img(img)  # (1,3,224,224)

        B = x.shape[0]

        # -------------------------
        # patch embedding (Conv1)
        # -------------------------
        x = self.visual.conv1(x)  # (B, 768, 14, 14)

        H, W = x.shape[-2:]

        x = x.reshape(B, self.embed_dim, -1)  # (B, 768, 196)
        x = x.permute(0, 2, 1)  # (B, 196, 768)

        # -------------------------
        # add CLS token
        # -------------------------
        cls = self.visual.class_embedding.to(x.dtype)
        cls = cls + torch.zeros(B, 1, self.embed_dim, device=x.device)

        x = torch.cat([cls, x], dim=1)  # (B,197,768)

        # positional embedding
        x = x + self.visual.positional_embedding.to(x.dtype)
        x = self.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # (seq, batch, dim)

        x = self.visual.transformer(x)

        x = x.permute(1, 0, 2)  # (B,197,768)

        # remove CLS token
        patch_tokens = x[:, 1:, :]  # (B,196,768)

        # reshape to spatial map
        spatial = patch_tokens.reshape(B, H, W, self.embed_dim)
        spatial = spatial.permute(0, 3, 1, 2).contiguous()

        return spatial, None
