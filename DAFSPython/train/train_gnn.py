import os
from datetime import datetime

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from util.npy_loader import EmbeddingDataset
from model.simple_gnn import  SimpleGNN

# 设置数据集和数据加载器
base_dir = "../../data/align/match"
print("Listing files in base_dir:", base_dir)
print(os.listdir(base_dir))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建训练集和测试集
train_dataset = EmbeddingDataset(base_dir, split='train', test_size=0.2, device=device)
test_dataset = EmbeddingDataset(base_dir, split='test', test_size=0.2, device=device)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 初始化模型和优化器
model = SimpleGNN(in_size=256, hidden_size=128, out_size=1).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()


# 创建存储模型记录的CSV文件路径
def create_csv_path():
    now = datetime.now().strftime("%Y-%m-%d")
    file_path = os.path.join("../../data/train", f"{now}.csv")
    return file_path


# 早停机制
class EarlyStopping:
    def __init__(self, patience=10, delta: float = 0.0):
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
                true_image_idx = label[i].item()

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


def train():
    # 训练循环
    early_stopping = EarlyStopping(patience=5, delta=0.001)  # 假设5个epoch没有改进则早停
    best_val_loss = float('inf')
    last_test_loss = float('inf')  # 保存上一次的 test_loss
    model_records = []

    for epoch in range(200):
        model.train()
        total_loss = 0
        for i, (text_vec, image_vec, label) in enumerate(train_dataloader):
            text_vec = text_vec.to(device)
            image_vec = image_vec.to(device)
            label = label.to(device)

            # 获取模型输出
            logits = model(text_vec, image_vec, label)

            logits = logits.view(-1)  # 将 logits 扁平化为一维张量
            label = label.view(-1)  # 将 label 扁平化为一维张量

            loss = criterion(logits, label.float())  # 计算损失

            optimizer.zero_grad()
            loss.backward()

            # 在反向传播后，更新前裁剪梯度
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item()

            # 添加调试信息，检查损失是否为 NaN
            if torch.isnan(loss):
                print(f"NaN detected at epoch {epoch}, batch {i}")
                print(f"Logits: {logits}")
                print(f"Label: {label}")
                return  # 终止训练，防止继续产生 NaN

        avg_loss = total_loss / len(train_dataloader)

        if epoch > 0 and epoch % 20 == 0:
            # 评估并记录
            model.eval()
            total_test_loss = 0
            with torch.no_grad():
                for text_vec, image_vec, label in test_dataloader:
                    text_vec = text_vec.to(device)
                    image_vec = image_vec.to(device)
                    label = label.to(device)

                    logits = model(text_vec, image_vec, label)
                    test_loss = criterion(logits.squeeze(), label.squeeze().float())
                    total_test_loss += test_loss.item()

            avg_test_loss = total_test_loss / len(test_dataloader)
            last_test_loss = avg_test_loss  # 保存供后续早停判断使用

            all_dataloader = DataLoader(
                train_dataset + test_dataset,  # 合并训练集和测试集
                batch_size=32,
                shuffle=False
            )

            # 计算 Hits@K
            hits_at_k = evaluate_hit_at_k(model, all_dataloader)

            # 打印评估结果
            print(f"[Eval @ Epoch {epoch}] Train Loss: {avg_loss:.4f}, Test Loss: {avg_test_loss:.4f}")
            for k, hits in hits_at_k.items():
                print(f"Hits@{k}: {hits:.4f}")

            # 记录每次评估的信息
            model_name = f"han_epoch{epoch}.pt"
            model_records.append({
                'epoch': epoch,
                'model_name': model_name,  # 只记录模型名称
                'Hits@1': hits_at_k[1],
                'Hits@5': hits_at_k[5],
                'Hits@10': hits_at_k[10],
                'train_loss': avg_loss,
                'test_loss': avg_test_loss,
                'timestamp': datetime.now().strftime("%Y-%m-%d")
            })

            # 判断是否保存最佳模型
            if avg_test_loss < best_val_loss:
                best_val_loss = avg_test_loss
                save_dir = "../gnn"
                os.makedirs(save_dir, exist_ok=True)
                model_path = os.path.join(save_dir, model_name)

                torch.save(model.state_dict(), model_path)  # 只保存最优模型

            # 非测试轮复用上一次的 test_loss
            else:
                avg_test_loss = last_test_loss

            # 每轮都执行早停判断
            if early_stopping(avg_test_loss):
                print("Early stopping triggered!")
                break

    return model_records

if __name__ == "__main__":
    records = train()
    print(records)