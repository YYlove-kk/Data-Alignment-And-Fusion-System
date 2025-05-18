import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleGNN(nn.Module):
    def __init__(self, in_size=256, hidden_size=128, out_size=1, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        # 共享的QKV投影（同质GNN不区分节点类型）
        self.q_proj = nn.Linear(in_size, hidden_size)
        self.k_proj = nn.Linear(in_size, hidden_size)
        self.v_proj = nn.Linear(in_size, hidden_size)

        # 注意力权重计算（同质注意力）
        self.attn_fc = nn.Linear(self.head_dim * 2, 1)

        # 输出层
        self.output_fc = nn.Linear(hidden_size * 3, out_size)

    def forward(self, text_vec, image_vec, label=None):

        B = text_vec.size(0)

        # 同质投影（文本和影像共享参数）
        Q = self.q_proj(torch.cat([text_vec, image_vec], dim=0))  # (2B, hidden_size)
        K = self.k_proj(torch.cat([text_vec, image_vec], dim=0))
        V = self.v_proj(torch.cat([text_vec, image_vec], dim=0))

        # 分头处理
        Q = Q.view(2*B, self.num_heads, self.head_dim)
        K = K.view(2*B, self.num_heads, self.head_dim)
        V = V.view(2*B, self.num_heads, self.head_dim)

        # 构建全连接边（模拟同质图）
        edge_index = []
        for i in range(B):
            edge_index.append([i, B+i])  # text_i -> image_i
            edge_index.append([B+i, i])  # image_i -> text_i
        edge_index = torch.tensor(edge_index).t().to(text_vec.device)  # (2, 2B)

        # 同质注意力计算
        attn_scores = []
        fused_features = []
        for h in range(self.num_heads):
            src, dst = edge_index
            q = Q[src, h, :]  # (2B, head_dim)
            k = K[dst, h, :]
            v = V[dst, h, :]

            # 注意力得分（无模态区分）
            qk = torch.cat([q, k], dim=-1)  # (2B, 2*head_dim)
            score = torch.sigmoid(self.attn_fc(qk))  # (2B, 1)

            # 聚合邻居（此处为1跳交互）
            fused = v * score  # (2B, head_dim)
            fused_features.append(fused)

        # 合并多头
        fused = torch.cat(fused_features, dim=-1)  # (2B, hidden_size)

        # 分离文本和影像特征
        text_fused = fused[:B]  # (B, hidden_size)
        image_fused = fused[B:]

        # 与原特征拼接后输出
        combined = torch.cat([
            text_fused + image_fused,  # 交互特征
            self.q_proj(text_vec),     # 原始文本特征
            self.q_proj(image_vec)      # 原始影像特征
        ], dim=-1)
        output = self.output_fc(combined)

        return output