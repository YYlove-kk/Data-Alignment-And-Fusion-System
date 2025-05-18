import os

import torch

import torch.nn as nn
from pkginfo.commandline import Simple
from torch.utils.data import DataLoader

from model.simple_han import SimpleHAN
from util.npy_loader import EmbeddingDataset
from model.model_han import AttentionHAN

from sklearn.metrics import roc_auc_score
# 设置数据集和数据加载器
base_dir = "../data/align/match"
print("Listing files in base_dir:", base_dir)
print(os.listdir(base_dir))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建训练集和测试集
train_dataset = EmbeddingDataset(base_dir, split='train', test_size=0.2, device=device)
test_dataset = EmbeddingDataset(base_dir, split='test', test_size=0.2, device=device)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 初始化模型和优化器
model = SimpleHAN(in_size=256, hidden_size=128, out_size=1).to(device)
model.load_state_dict(torch.load("simplehan/han_epoch60.pt", map_location=device))
model.eval()  # 设置为评估模式
criterion = nn.BCEWithLogitsLoss()

def evaluate_hit_at_k(model, test_loader):
    model.eval()
    k_values = [1, 5, 10]
    hits_at_k: dict[int, float] = {k: 0.0 for k in k_values}  # 初始化命中计数器

    with torch.no_grad():
        # 遍历测试集
        for text_vec, image_vec, label in test_loader:
            batch_size = label.size(0)
            for i in range(batch_size):
                # 取出单个样本
                text = text_vec[i]
                img = image_vec[i]

                # 对该文本计算与整个 batch 内所有图像的相似度
                scores = []
                for idx, img2 in enumerate(image_vec):
                    score = model(text.unsqueeze(0), img2.unsqueeze(0))
                    prob = torch.sigmoid(score).item()
                    scores.append((idx, prob))

                # 排序并计算 top-k 是否命中
                sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
                for k in k_values:
                    top_k = sorted_scores[:k]
                    if any(idx == i for idx, _ in top_k):  # i 是当前文本对应的真实图像索引
                        hits_at_k[k] += 1

    # 计算Hits@K的准确率并返回
    for k in k_values:
        hits_at_k[k] = float(hits_at_k[k]) / len(test_loader.dataset)  # 平均值，确保转换为 float

    return hits_at_k

def evaluate_auc(model, dataloader):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for text_vec, image_vec, label in dataloader:
            text_vec = text_vec.to(device)
            image_vec = image_vec.to(device)
            label = label.to(device)

            logits = model(text_vec, image_vec, label)
            probs = torch.sigmoid(logits).view(-1).cpu().numpy()  # 转为概率

            all_probs.extend(probs)
            all_labels.extend(label.view(-1).cpu().numpy())

    auc = roc_auc_score(all_labels, all_probs)
    return auc

model.eval()
all_dataloader = DataLoader(
    train_dataset + test_dataset,  # 合并训练集和测试集
    batch_size=16,
    shuffle=False
)

# 计算 Hits@K
hits_at_k = evaluate_hit_at_k(model, all_dataloader)
for k, hits in hits_at_k.items():
    print(f"Hits@{k}: {hits:.4f}")

auc_score = evaluate_auc(model, all_dataloader)
print(f"AUC: {auc_score:.4f}")