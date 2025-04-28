import dgl
import torch
import torch.nn as nn
import dgl.nn as dglnn

class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)
        beta = torch.softmax(w, dim=0)
        return (beta.unsqueeze(0) * z).sum(1)

class HANLayer(nn.Module):
    def __init__(self, in_size, out_size, num_heads, meta_paths):
        super().__init__()
        self.gat_layers = nn.ModuleList([
            dglnn.GATConv(in_size, out_size, num_heads) for _ in meta_paths
        ])
        self.semantic_attention = SemanticAttention(out_size * num_heads)
        self.meta_paths = meta_paths
        self.cached_graphs = {}

    def forward(self, g, h):
        if not self.cached_graphs:
            for meta_path in self.meta_paths:
                self.cached_graphs[tuple(meta_path)] = dgl.metapath_reachable_graph(g, meta_path)
        out = []
        for i, meta_path in enumerate(self.meta_paths):
            new_g = self.cached_graphs[tuple(meta_path)]
            out.append(self.gat_layers[i](new_g, h).flatten(1))
        out = torch.stack(out, dim=1)
        return self.semantic_attention(out)

class HAN(nn.Module):
    def __init__(self, meta_paths, in_size=512, hidden_size=128, out_size=1, num_heads=8):
        super().__init__()
        self.han_layer = HANLayer(in_size, hidden_size, num_heads, meta_paths)
        self.predict = nn.Linear(hidden_size, out_size)

    def forward(self, g, h_dict):
        h = self.han_layer(g, h_dict['Visit'].float())
        return {'Visit': self.predict(h)}
