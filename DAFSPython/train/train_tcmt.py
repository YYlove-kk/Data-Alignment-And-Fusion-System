import argparse

import torch
import numpy as np
import glob
import random
import os
from torch.utils.data import Dataset, DataLoader
from model.model_tcmt import TCMT, clip_loss
from sklearn.metrics.pairwise import cosine_similarity
import json


class PairDataset(Dataset):
    def __init__(self, txt_dir, img_dir, split=1, train=True):
        ids = [os.path.basename(p).replace("_txt.npy", "")
               for p in glob.glob(f"{txt_dir}/*_txt.npy")]
        random.shuffle(ids)
        border = int(len(ids) * split)
        self.ids = ids[:border] if train else ids[border:]
        self.txt_dir = txt_dir
        self.img_dir = img_dir

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        pid = self.ids[idx]
        txt = np.load(f"{self.txt_dir}/{pid}_txt.npy")
        img = np.load(f"{self.img_dir}/{pid}_img.npy")

        return torch.tensor(txt, dtype=torch.float32), \
            torch.tensor(img, dtype=torch.float32)


# 定义保存 embeddings 的函数
def save_embeddings_per_patient(epoch, z_t, z_i, output_dir, patient_ids):
    for idx, patient_id in enumerate(patient_ids):
        # 按患者 ID 保存 embeddings
        np.save(f"{output_dir}/{patient_id}_z_t_epoch{epoch}.npy", z_t[idx].cpu().detach().numpy())
        np.save(f"{output_dir}/{patient_id}_z_i_epoch{epoch}.npy", z_i[idx].cpu().detach().numpy())


def calculate_semantic_accuracy(similarity_matrix, threshold=0.7):
    """
    计算语义准确率：相似度大于阈值的对认为是正确的对齐。
    """
    # 获取相似度大于阈值的正确对数
    correct_alignments = (similarity_matrix > threshold).sum()
    total_pairs = similarity_matrix.size
    accuracy = correct_alignments / total_pairs  # 准确率 = 正确对数 / 总对数
    return accuracy


def calculate_alignment_coverage(similarity_matrix, threshold=0.7):
    """
    计算对齐覆盖数：相似度大于阈值的对的数量。
    """
    covered_pairs = (similarity_matrix > threshold).sum()  # 统计符合条件的对的数量
    return covered_pairs

def save_alignment_matrix(zt, zi, patient_ids, output_dir):
    """
    计算对齐矩阵，保存alignment_matrix.json和diagonal_similarity.json
    """
    similarity_matrix = cosine_similarity(zt.cpu().detach().numpy(), zi.cpu().detach().numpy())

    # 计算语义准确率和对齐覆盖数
    accuracy = calculate_semantic_accuracy(similarity_matrix)
    coverage = calculate_alignment_coverage(similarity_matrix)

    # 提取自己对应自己的相似度（取对角线）
    diagonal = np.diag(similarity_matrix)

    # 保存 diagonal_similarity.json
    similarity_dict = {patient_id: float(sim) for patient_id, sim in zip(patient_ids, diagonal)}
    with open(os.path.join(output_dir, "diagonal_similarity.json"), "w") as f:
        json.dump(similarity_dict, f, indent=2)

    # 准备 alignment_result 内容
    alignment_result = {
        "alignment_matrix": similarity_matrix.tolist(),
        "semantic_accuracy": accuracy,
        "alignment_coverage": coverage,
        "diagonal_similarity": diagonal.tolist(),
        "patient_ids": patient_ids
    }

    return alignment_result


# 定义主函数
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--txt_dir", required=True)
    parser.add_argument("--img_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    # 设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建 Dataset 和 DataLoader
    ds = PairDataset(args.txt_dir, args.img_dir, split=1, train=True)
    dl = DataLoader(ds, batch_size=64, shuffle=True)

    # 初始化模型和优化器
    net = TCMT().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4)

    # 训练循环
    for epoch in range(20):
        net.train()
        all_z_t = []
        all_z_i = []

        for txt, img in dl:
            # 把输入数据放到设备上
            txt = txt.to(device)
            img = img.to(device)
            zt, zi = net(txt, img)

            loss = clip_loss(zt, zi, net.logit_scale.exp())
            loss.backward()
            opt.step()
            opt.zero_grad()

            # 记录每一批次的 z_t 和 z_i
            all_z_t.append(zt.detach().cpu())  # 存回CPU，防止显存爆掉
            all_z_i.append(zi.detach().cpu())

        # 将每一批次的结果拼接起来
        all_z_t = torch.cat(all_z_t, dim=0)
        all_z_i = torch.cat(all_z_i, dim=0)

        # 保存每个 epoch 的 embeddings
        save_embeddings_per_patient(epoch, all_z_t, all_z_i, args.output_dir, ds.ids)

        # 计算并保存对齐矩阵，同时保存 diagonal_similarity.json
        alignment_result = save_alignment_matrix(all_z_t, all_z_i, ds.ids, args.output_dir)

        # 输出对齐矩阵为 JSON 格式
        print(json.dumps(alignment_result))

        # 保存模型
        torch.save(net.state_dict(), f"{args.output_dir}/tcmt_ep{epoch}.pt")


if __name__ == "__main__":
    main()
