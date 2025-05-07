import os
from datetime import datetime

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from util.npy_loader import EmbeddingDataset
from model.model_han import AttentionHAN

# 设置数据集和数据加载器
base_dir = "data/align/reduce"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建训练集和测试集
train_dataset = EmbeddingDataset(base_dir, split='train', test_size=0.2, device=device)
test_dataset = EmbeddingDataset(base_dir, split='test', test_size=0.2, device=device)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 初始化模型和优化器
model = AttentionHAN(in_size=512, hidden_size=128, out_size=1).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()

# 计算 Hits@N 的函数
def calculate_hits_at_n(model, test_dataloader, k_values=[1, 5, 10], device=None):
    model.eval()
    hits_at_k = {k: 0 for k in k_values}
    total = 0

    with torch.no_grad():
        for text_vec, image_vec, label in test_dataloader:
            text_vec = text_vec.to(device)
            image_vec = image_vec.to(device)

            # 计算模型的输出
            logits = model(text_vec, image_vec)

            # 排序模型输出并找到前 N 个预测
            _, top_k_idx = torch.topk(logits, max(k_values), dim=1, largest=True, sorted=True)
            label = label.to(device)

            # 计算 Hits@K
            for k in k_values:
                hits_at_k[k] += torch.sum(top_k_idx[:, :k] == label.view(-1, 1)).item()

            total += text_vec.size(0)

    # 计算平均的 Hits@K
    hits_at_k = {k: hits / total for k, hits in hits_at_k.items()}

    return hits_at_k

# 创建存储模型记录的CSV文件路径
def create_csv_path():
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_path = os.path.join("data/train", f"{now}.csv")
    return file_path

# 早停机制
class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop

# 训练循环
early_stopping = EarlyStopping(patience=5, delta=0.001)  # 假设5个epoch没有改进则早停
best_val_loss = float('inf')

# 初始化 CSV 记录
csv_file_path = create_csv_path()
model_records = []

for epoch in range(100):
    model.train()
    total_loss = 0
    for text_vec, image_vec, label in train_dataloader:
        # 计算预测值
        logits = model(text_vec, image_vec)
        loss = criterion(logits, label)

        # 更新模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_dataloader)

    # 在测试集上评估并计算 Hits@N
    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for text_vec, image_vec, label in test_dataloader:
            logits = model(text_vec, image_vec)
            test_loss = criterion(logits, label)
            total_test_loss += test_loss.item()

    avg_test_loss = total_test_loss / len(test_dataloader)

    # 计算 Hits@K
    hits_at_k = calculate_hits_at_n(model, test_dataloader, k_values=[1, 5, 10], device=device)

    # 输出结果
    print(f"Epoch {epoch}, Train Loss: {avg_loss:.4f}, Test Loss: {avg_test_loss:.4f}")
    for k, hits in hits_at_k.items():
        print(f"Hits@{k}: {hits:.4f}")

    # 早停检查
    if avg_test_loss < best_val_loss:
        best_val_loss = avg_test_loss
        # 保存最佳模型
        save_dir = "DAFSPython/han"
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, f"han_epoch{epoch}.pt")
        torch.save(model.state_dict(), model_path)

        # 保存模型记录
        model_records.append({
            'epoch': epoch,
            'model_path': model_path,
            'Hits@1': hits_at_k[1],
            'Hits@5': hits_at_k[5],
            'Hits@10': hits_at_k[10],
            'train_loss': avg_loss,
            'test_loss': avg_test_loss,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    else:
        if early_stopping(avg_test_loss):
            print("Early stopping triggered!")
            break

