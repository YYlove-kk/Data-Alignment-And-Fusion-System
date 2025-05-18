import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleHAN(nn.Module):
    def __init__(self, in_size=256, hidden_size=128, out_size=1, threshold=0.6):
        super().__init__()
        self.hidden_size = hidden_size
        self.threshold = threshold

        # QKV projection for text and image
        self.text_q_proj = nn.Linear(in_size, hidden_size)
        self.text_k_proj = nn.Linear(in_size, hidden_size)
        self.text_v_proj = nn.Linear(in_size, hidden_size)

        self.image_q_proj = nn.Linear(in_size, hidden_size)
        self.image_k_proj = nn.Linear(in_size, hidden_size)
        self.image_v_proj = nn.Linear(in_size, hidden_size)

        # Attention scoring layer (single head)
        self.attn_weight = nn.Linear(hidden_size * 2, 1)

        # Final output projection
        self.output_fc = nn.Linear(hidden_size * 3, out_size)

    def compute_confidence(self, features, label):

        with torch.no_grad():
            # features: (B, hidden_size)
            # label: (B,)
            B, D = features.shape
            label_expand = label.view(-1, 1).repeat(1, D).float()  # (B, D)
            corr = torch.abs(features - label_expand).mean(dim=0)  # (D,)
            confidence = 1.0 - corr / (corr.max() + 1e-6)  # (D,)，归一化为0~1
        return confidence  # (D,)

    def forward(self, text_vec, image_vec, label=None):
        """
        Inputs:
            text_vec:  (B, in_size)
            image_vec: (B, in_size)
            label:     (B,), optional
        """
        B = text_vec.size(0)

        # QKV projection
        t_Q = self.text_q_proj(text_vec)  # (B, hidden_size)
        t_K = self.text_k_proj(text_vec)  # (B, hidden_size)
        t_V = self.text_v_proj(text_vec)  # (B, hidden_size)

        i_Q = self.image_q_proj(image_vec)  # (B, hidden_size)
        i_K = self.image_k_proj(image_vec)  # (B, hidden_size)
        i_V = self.image_v_proj(image_vec)  # (B, hidden_size)

        # attention score
        qk_pair = torch.cat([t_Q, i_K], dim=-1)  # (B, 2 * hidden_size)
        attn_score = torch.sigmoid(self.attn_weight(qk_pair))  # (B, 1)

        if self.training and label is not None:
            # 训练模式下，根据label计算置信度，调整注意力权重
            confidence_t = self.compute_confidence(t_V, label).unsqueeze(0)  # (1, hidden_size)
            confidence_i = self.compute_confidence(i_V, label).unsqueeze(0)  # (1, hidden_size)

            # 将attn_score扩展到hidden_size维度
            attn_score_expand = attn_score.expand(-1, self.hidden_size)  # (B, hidden_size)

            # 权重加权融合，举例简单加权
            t_selected = t_V * attn_score_expand * confidence_t
            i_selected = i_V * (1 - attn_score_expand) * confidence_i
        else:
            # 推理时直接用注意力权重加权融合
            t_selected = t_V * attn_score
            i_selected = i_V * (1 - attn_score)

        fused = t_selected + i_selected  # (B, hidden_size)

        # Final concat and output
        combined = torch.cat([fused, t_Q, i_Q], dim=-1)  # (B, hidden_size * 3)
        output = self.output_fc(combined)  # (B, out_size)

        return output
