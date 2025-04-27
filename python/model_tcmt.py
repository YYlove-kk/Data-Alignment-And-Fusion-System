import torch
import torch.nn as nn


class TCMT(nn.Module):
    def __init__(self, d_in_txt=800, d_in_img=2080, d_model=512, heads=8):
        super().__init__()
        self.text_proj = nn.Linear(d_in_txt, d_model)
        self.img_proj  = nn.Linear(d_in_img, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=2048,
            batch_first=True
        )

        self.xattn_t2i = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.xattn_i2t = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.logit_scale = nn.Parameter(torch.ones([]) * (1 / 0.07))

    def forward(self, txt, img):
        # txt/img shape: (B, T, d_in)
        z_t = nn.functional.normalize(self.text_proj(txt), dim=-1)  # (B, T, 512)
        z_i = nn.functional.normalize(self.img_proj(img), dim=-1)  # (B, T, 512)

        # Cross attention without unsqueeze
        z_t = self.xattn_t2i(z_t)  # (B, T, 512)
        z_i = self.xattn_i2t(z_i)  # (B, T, 512)

        # 聚合方式：mean pooling
        z_t = z_t.mean(dim=1)  # (B, 512)
        z_i = z_i.mean(dim=1)  # (B, 512)

        return z_t, z_i


def clip_loss(z_t, z_i, logit_scale):
    logits = logit_scale * z_t @ z_i.T  # (B, B)
    labels = torch.arange(len(z_t)).to(z_t.device)
    loss_t = nn.functional.cross_entropy(logits, labels)
    loss_i = nn.functional.cross_entropy(logits.T, labels)
    return (loss_t + loss_i) / 2
