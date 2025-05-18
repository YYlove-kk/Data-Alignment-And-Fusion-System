import torch
import torch.nn as nn
import torch.nn.functional as F


class TCMT(nn.Module):
    def __init__(self, d_in_txt=800, d_in_img=2080, d_model=512, heads=8):
        super().__init__()

        # 线性投影到公共维度
        self.text_proj = nn.Linear(d_in_txt, d_model)
        self.img_proj  = nn.Linear(d_in_img, d_model)

        # Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=2048,
            batch_first=True
        )

        # 分别为 text->image 和 image->text 建立 cross-attention 编码器
        self.xattn_t2i = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.xattn_i2t = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 可学习温度参数（用于对比学习）
        self.logit_scale = nn.Parameter(torch.ones([]) * (1 / 0.07))

    def forward(self, txt, img):
        """
        txt: shape (B, 1, 800)
        img: shape (B, N, 2080)
        returns:
            z_t: (B, 512)
            z_i: (B, 512)
        """
        # 投影 + 归一化
        z_t = F.normalize(self.text_proj(txt), dim=-1)  # (B, 1, 512)
        z_i = F.normalize(self.img_proj(img), dim=-1)   # (B, N, 512)

        # 编码（注意 TransformerEncoder 不改变序列长度）
        z_t = self.xattn_t2i(z_t)  # (B, 1, 512)
        z_i = self.xattn_i2t(z_i)  # (B, N, 512)

        # 聚合方式：mean pooling
        z_t = z_t.mean(dim=1)  # (B, 512)
        z_i = z_i.mean(dim=1)  # (B, 512)

        return z_t, z_i


def clip_loss(z_t, z_i, logit_scale):
    """
    对比损失函数（CLIP 风格）
    z_t, z_i: shape (B, 512)
    logit_scale: 可学习温度参数
    """
    logits = logit_scale * z_t @ z_i.T  # (B, B)
    labels = torch.arange(len(z_t)).to(z_t.device)
    loss_t = F.cross_entropy(logits, labels)
    loss_i = F.cross_entropy(logits.T, labels)
    return (loss_t + loss_i) / 2
