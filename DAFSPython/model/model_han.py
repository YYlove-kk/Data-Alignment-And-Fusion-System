import torch
import torch.nn as nn

class AttentionHAN(nn.Module):
    def __init__(self, in_size=256, hidden_size=128, out_size=1, num_heads=4, threshold=0.7):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.threshold = threshold

        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        # QKV projection for text and image
        self.text_q_proj = nn.Linear(in_size, hidden_size)
        self.text_k_proj = nn.Linear(in_size, hidden_size)
        self.text_v_proj = nn.Linear(in_size, hidden_size)

        self.image_q_proj = nn.Linear(in_size, hidden_size)
        self.image_k_proj = nn.Linear(in_size, hidden_size)
        self.image_v_proj = nn.Linear(in_size, hidden_size)

        # Per-head attention scoring
        self.attn_weights = nn.ModuleList([
            nn.Linear(self.head_dim * 2, 1) for _ in range(num_heads)
        ])

        # Final output projection
        self.output_fc = nn.Linear(hidden_size * 3, out_size)

    def compute_chi_square(self, feature, label, num_classes=2):

        chi_squared = torch.zeros(feature.shape[1]).to(feature.device)

        for f in range(feature.shape[1]):
            freq_table = torch.zeros((num_classes, num_classes)).to(feature.device)
            for i in range(feature.shape[0]):
                f_val = int(feature[i, f].item() > self.threshold)
                l_val = int(label[i].item())
                freq_table[f_val, l_val] += 1

            total = torch.sum(freq_table)
            expected = (torch.sum(freq_table, dim=0).view(-1, 1) * torch.sum(freq_table, dim=1).view(1, -1)) / (total + 1e-6)
            observed = freq_table
            chi_squared[f] = torch.sum((observed - expected) ** 2 / (expected + 1e-6))

        return chi_squared  # (D,)

    def forward(self, text_vec, image_vec, label=None):
        """
        Inputs:
            text_vec:  (B, in_size)
            image_vec: (B, in_size)
            label:     (B,), optional
        """
        B = text_vec.size(0)

        # QKV projection
        t_Q = self.text_q_proj(text_vec).view(B, self.num_heads, self.head_dim)
        t_K = self.text_k_proj(text_vec).view(B, self.num_heads, self.head_dim)
        t_V = self.text_v_proj(text_vec).view(B, self.num_heads, self.head_dim)

        i_Q = self.image_q_proj(image_vec).view(B, self.num_heads, self.head_dim)
        i_K = self.image_k_proj(image_vec).view(B, self.num_heads, self.head_dim)
        i_V = self.image_v_proj(image_vec).view(B, self.num_heads, self.head_dim)

        if self.training and label is not None:
            # Training mode: compute chi-square confidence
            chi_t = self.compute_chi_square(t_V.view(B, -1), label)
            chi_i = self.compute_chi_square(i_V.view(B, -1), label)
            chi_max = torch.max(torch.max(chi_t), torch.max(chi_i))
            alpha_t = chi_t / (chi_max + 1e-6)  # (hidden_size,)
            alpha_i = chi_i / (chi_max + 1e-6)
        else:
            # Inference mode: use default uniform confidence
            alpha_t = torch.ones(self.num_heads).to(text_vec.device)
            alpha_i = torch.ones(self.num_heads).to(text_vec.device)

        fused_heads = []

        for h in range(self.num_heads):
            t_q = t_Q[:, h, :]  # (B, head_dim)
            t_k = t_K[:, h, :]
            t_v = t_V[:, h, :]

            i_q = i_Q[:, h, :]
            i_k = i_K[:, h, :]
            i_v = i_V[:, h, :]

            # attention score
            qk_pair = torch.cat([t_q, i_k], dim=-1)  # (B, 2 * head_dim)
            attn_score = torch.sigmoid(self.attn_weights[h](qk_pair))  # (B, 1)

            # weight * confidence
            e_t = attn_score * alpha_t[h]
            e_i = attn_score * alpha_i[h]

            t_selected = t_v * e_t
            i_selected = i_v * e_i

            # weighted fusion
            fused = t_selected + (1 - e_t) * i_selected
            fused_heads.append(fused)

        # (B, hidden_size)
        fused = torch.cat(fused_heads, dim=-1)

        # Final concat and output
        t_q_flat = self.text_q_proj(text_vec)  # (B, hidden_size)
        i_q_flat = self.image_q_proj(image_vec)  # (B, hidden_size)

        combined = torch.cat([fused, t_q_flat, i_q_flat], dim=-1)  # (B, hidden_size * 3)
        output = self.output_fc(combined)  # (B, out_size)

        return output
