import sys

from jsonargparse.typing import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms.functional as TF


from dinov3.data.transforms import make_classification_eval_transform
from dinov3.hub.dinotxt import dinov3_vitl16_dinotxt_tet1280d20h24l


# =========================
# 1. Cost Aggregation (CAT-Seg style)
# =========================
class CostAggregation(nn.Module):
    def __init__(self, dim=768):
        super().__init__()

        self.image_proj = nn.Linear(dim, dim)
        self.text_proj = nn.Linear(dim, dim)

        self.spatial_attn = nn.MultiheadAttention(dim, 8, batch_first=True)
        self.class_attn = nn.MultiheadAttention(dim, 8, batch_first=True)

        self.fuse = nn.Linear(dim * 2, dim)

    def forward(self, img_feat, txt_feat):
        B, N, D = img_feat.shape
        C = txt_feat.shape[1]

        img = F.normalize(self.image_proj(img_feat), dim=-1)
        txt = F.normalize(self.text_proj(txt_feat), dim=-1)

        # cost volume
        cost = torch.bmm(img, txt.transpose(1, 2))  # [B, N, C]

        # spatial aggregation
        spatial = cost.reshape(B, N * C, 1).repeat(1, 1, D)
        spatial, _ = self.spatial_attn(spatial, spatial, spatial)
        spatial = spatial.reshape(B, N, C, D).mean(2)

        # class aggregation
        cls = cost.transpose(1, 2).reshape(B, C * N, 1).repeat(1, 1, D)
        cls, _ = self.class_attn(cls, cls, cls)
        cls = cls.reshape(B, C, N, D).mean(2)

        spatial = spatial.unsqueeze(2).expand(-1, -1, C, -1)
        cls = cls.unsqueeze(1).expand(-1, N, -1, -1)

        fused = torch.cat([spatial, cls], dim=-1)
        fused = self.fuse(fused)

        return fused.squeeze(-1)  # [B, N, C]


# =========================
# 2. Model
# =========================
class DinoCATSeg(nn.Module):
    def __init__(self, model, tokenizer):
        super().__init__()

        self.model = model
        self.tokenizer = tokenizer

        self.cost = CostAggregation(768)

        self.text_embed = nn.Parameter(torch.randn(10, 768) * 0.02)

    def encode_image(self, x):
        _, patch, _ = self.model.encode_image_with_patch_tokens(x)
        return patch

    def forward(self, x, class_names=None, use_learned_text=False):
        B, _, H, W = x.shape

        img_feat = self.encode_image(x)

        if use_learned_text:
            txt_feat = self.text_embed.unsqueeze(0).expand(B, -1, -1)
        else:
            tokens = self.tokenizer.tokenize(class_names).to(x.device)
            txt_feat = self.model.encode_text(tokens)
            txt_feat = txt_feat[:, 1024:]
            txt_feat = txt_feat.unsqueeze(0).expand(B, -1, -1)

        cost_feat = self.cost(img_feat, txt_feat)

        Ht = Wt = int(cost_feat.shape[1] ** 0.5)

        feat = cost_feat.permute(0, 2, 1).reshape(B, -1, Ht, Wt)

        return F.interpolate(feat, size=(H, W), mode="bilinear", align_corners=False)


# =========================
# 3. TTA (FIXED VERSION)
# =========================
class TTA:
    def __init__(self, model, lr=1e-4):
        self.model = model

        self.opt = torch.optim.Adam([
            {"params": model.cost.parameters()},
            {"params": model.text_embed}
        ], lr=lr)

    # -------- strong augmentation --------
    def aug(self, x):
        if torch.rand(1) > 0.5:
            x = TF.hflip(x)

        if torch.rand(1) > 0.5:
            x = TF.adjust_brightness(x, 0.7 + torch.rand(1).item() * 0.6)

        if torch.rand(1) > 0.5:
            x = TF.adjust_contrast(x, 0.7 + torch.rand(1).item() * 0.6)

        return x

    # -------- FIXED LOSS --------
    def loss(self, preds):

        loss = 0.0

        for p in preds:
            prob = F.softmax(p, dim=1)

            # 1. entropy minimization (TENT core)
            entropy = -(prob * torch.log(prob + 1e-8)).sum(1).mean()
            loss += 0.1 * entropy

            # 2. pseudo-label sharpening
            pseudo = prob ** 2
            pseudo = pseudo / (pseudo.sum(dim=1, keepdim=True) + 1e-8)

            loss += F.kl_div(
                prob.log(),
                pseudo.detach(),
                reduction="batchmean"
            )

        # 3. consistency between views
        for i in range(len(preds)):
            for j in range(i + 1, len(preds)):
                loss += F.mse_loss(preds[i], preds[j])

        return loss

    # -------- adapt --------
    def adapt(self, x, class_names, steps=5):
        self.model.train()

        for _ in range(steps):
            self.opt.zero_grad()

            preds = []

            for _ in range(3):
                aug = self.aug(x)
                preds.append(
                    self.model(aug, class_names, use_learned_text=True)
                )

            loss = self.loss(preds)
            loss.backward()
            self.opt.step()

        self.model.eval()

        with torch.no_grad():
            return self.model(x, class_names, use_learned_text=True)


# =========================
# 4. Main
# =========================
def main():

    model, tokenizer = dinov3_vitl16_dinotxt_tet1280d20h24l()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    seg = DinoCATSeg(model, tokenizer).to(device)
    tta = TTA(seg)

    image = Image.open("asset/9.jpg").convert("RGB")

    transform = make_classification_eval_transform()
    x = transform(image).unsqueeze(0).to(device)

    class_names = ["road", "building", "tree", "water", "background"]

    print("Running TTA...")
    out = tta.adapt(x, class_names)

    mask = out.argmax(dim=1)[0].cpu().numpy()

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap="jet")
    plt.axis("off")

    plt.show()


if __name__ == "__main__":
    main()