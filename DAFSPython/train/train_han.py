import torch
import torch.nn as nn
import torch.optim as optim
from model.model_han import HAN
from util.dgl_loader import load_neo4j

g = load_neo4j("bolt://localhost:7687")

meta_paths = [
    [('Visit', 'SIMILAR', 'Visit')],
    [('Visit', 'BELONGS_TO', 'Source'), ('Source', 'BELONGS_TO', 'Visit')]
]

model = HAN(meta_paths).cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()

for epoch in range(100):
    h_dict = {ntype: g.nodes[ntype].data['feat'].cuda() for ntype in g.ntypes}
    y = g.nodes['Visit'].data['label'].float().cuda()
    logits = model(g, h_dict)['Visit'].squeeze()
    loss = criterion(logits, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch} Loss: {loss.item():.4f}")
        torch.save(model.state_dict(), f"han_epoch{epoch}.pt")
