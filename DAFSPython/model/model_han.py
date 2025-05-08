import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionHAN(nn.Module):
    def __init__(self, in_size=256, hidden_size=128, out_size=1, num_heads=4, threshold=0.6):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.threshold = threshold

        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        # 映射输入向量到统一空间
        self.text_proj = nn.Linear(in_size, hidden_size)
        self.image_proj = nn.Linear(in_size, hidden_size)

        # 多头注意力参数：每个头一个权重映射
        self.attn_weights = nn.ModuleList([
            nn.Linear(self.head_dim * 2, 1) for _ in range(num_heads)
        ])

        # 融合 + 输出层
        self.output_fc = nn.Linear(hidden_size * 3, out_size)

    def forward(self, text_vec, image_vec):
        # Step 1: 输入映射并 reshape 为多头结构
        t_feat = self.text_proj(text_vec)  # (B, hidden)
        i_feat = self.image_proj(image_vec)

        t_feat = t_feat.view(-1, self.num_heads, self.head_dim)  # (B, H, D)
        i_feat = i_feat.view(-1, self.num_heads, self.head_dim)

        # Step 2: 多头注意力权重计算与特征筛选
        fused_heads = []
        for i in range(self.num_heads):
            t_i = t_feat[:, i, :]  # (B, D)
            i_i = i_feat[:, i, :]
            combined_i = torch.cat([t_i, i_i], dim=-1)  # (B, 2D)
            attn_score = torch.sigmoid(self.attn_weights[i](combined_i))  # (B, 1)

            # 特征筛选
            t_mask = attn_score > self.threshold
            i_mask = attn_score > self.threshold

            t_i_selected = t_i * t_mask.float()
            i_i_selected = i_i * i_mask.float()

            fused = attn_score * t_i_selected + (1 - attn_score) * i_i_selected  # (B, D)
            fused_heads.append(fused)

        # Step 3: 融合所有头
        fused = torch.cat(fused_heads, dim=-1)  # (B, hidden)

        # Step 4: 拼接所有特征用于最终预测
        combined = torch.cat([fused, self.text_proj(text_vec), self.image_proj(image_vec)], dim=-1)  # (B, hidden*3)
        output = self.output_fc(combined)  # (B, 1)

        return output

