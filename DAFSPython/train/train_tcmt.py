import argparse
from datetime import datetime
import pandas as pd
import torch
import numpy as np
import glob
import random
import os

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from model.model_tcmt import TCMT, clip_loss
import json

def extract_date_from_filename(name):
    for part in name.split("_"):
        if part.isdigit() and len(part) == 14:
            return datetime.strptime(part, "%Y%m%d%H%M%S")
    return None

def extract_patient_name_from_filename(name):
    parts = name.split("_")
    if len(parts) > 1:
        return parts[0]
    return None

def load_npy_with_date(npy_dir):
    data = []
    for path in glob.glob(os.path.join(npy_dir, "*.npy")):
        date = extract_date_from_filename(os.path.basename(path))
        if date:
            data.append((os.path.basename(path), date))
    return data

def match_by_date(text_data, image_data):
    matched = []
    for txt_file, txt_date, txt_vec in text_data:
        best_match = None
        min_diff = float('inf')
        for img_file, img_date, img_vec in image_data:
            diff = abs((txt_date - img_date).days)
            if diff < min_diff:
                min_diff = diff
                best_match = (txt_file, img_file, txt_vec, img_vec)
        if best_match:
            matched.append(best_match)
    return matched

def get_patient_ids(text_dir,image_dir,match_dir):
    os.makedirs(match_dir, exist_ok=True)

    print("加载嵌入...")
    text_data = load_npy_with_date(text_dir)
    image_data = load_npy_with_date(image_dir)

    print(f"表格数: {len(text_data)}, 影像数: {len(image_data)}")

    print("匹配中...")
    matches = match_by_date(text_data, image_data)
    print(f"匹配对数: {len(matches)}")

    index = []

    for i, (txt_file, img_file) in enumerate(tqdm(matches)):

        patient_id = extract_patient_name_from_filename(img_file)
        index.append({
            "patient_id": patient_id,
            "text_file": txt_file,
            "image_file": img_file
        })

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    match_index_file = os.path.join(match_dir, f"match_index_{timestamp}.csv")

    df = pd.DataFrame(index)
    df.to_csv(match_index_file, index=False)
    print(f"✅ 匹配完成，保存至 {match_index_file}")
    return index

class PairDataset(Dataset):
    def __init__(self, match_index, txt_dir, img_dir, split=1, train=True):
        random.shuffle(match_index)
        border = int(len(match_index) * split)
        self.data = match_index[:border] if train else match_index[border:]
        self.txt_dir = txt_dir
        self.img_dir = img_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        txt_path = os.path.join(self.txt_dir, item["text_file"])
        img_path = os.path.join(self.img_dir, item["image_file"])
        txt = torch.from_numpy(np.load(txt_path)).float()
        img = torch.from_numpy(np.load(img_path)).float()


        return torch.tensor(txt, dtype=torch.float32), \
            torch.tensor(img, dtype=torch.float32), \
            item["patient_id"]



# 保存 embeddings
def save_embeddings_per_patient(z_t, z_i, output_dir, patient_ids):
    for idx, patient_id in enumerate(patient_ids):
        # 只保存最后一个 epoch 的嵌入向量
            np.save(f"{output_dir}/{patient_id}_z_t.npy", z_t[idx].cpu().detach().numpy())
            np.save(f"{output_dir}/{patient_id}_z_i.npy", z_i[idx].cpu().detach().numpy())



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

def calculate_alignment_matrix(zt, zi, patient_ids):
    """
    计算对齐矩阵
    """
    similarity_matrix = cosine_similarity(zt.cpu().detach().numpy(), zi.cpu().detach().numpy())

    # 计算语义准确率和对齐覆盖数
    accuracy = calculate_semantic_accuracy(similarity_matrix)
    coverage = calculate_alignment_coverage(similarity_matrix)

    # 提取自己对应自己的相似度（取对角线）
    diagonal = np.diag(similarity_matrix)

    # 准备 alignment_result 内容
    alignment_result = {
        "alignment_matrix": similarity_matrix.tolist(),
        "semantic_accuracy": accuracy,
        "alignment_coverage": coverage,
        "diagonal_similarity": diagonal.tolist(),
        "patient_ids": patient_ids
    }

    return alignment_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    text_dir = "data/align/text"
    image_dir = "data/align/image"
    match_dir = "data/align/output/match"
    args = parser.parse_args()

    # 设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    match_index = get_patient_ids(text_dir, image_dir, match_dir)
    ds = PairDataset(match_index, text_dir, image_dir, split=1, train=True)
    dl = DataLoader(ds, batch_size=64, shuffle=True)

    # 初始化模型和优化器
    net = TCMT().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4)

    # 训练循环
    for epoch in range(20):
        net.train()
        all_z_t = []
        all_z_i = []
        all_patient_ids = []

        for txt, img, pid in dl:
            txt = txt.to(device)
            img = img.to(device)
            zt, zi = net(txt, img)

            loss = clip_loss(zt, zi, net.logit_scale.exp())
            loss.backward()
            opt.step()
            opt.zero_grad()

            all_z_t.append(zt.detach().cpu())
            all_z_i.append(zi.detach().cpu())
            all_patient_ids.extend(pid)

        # 仅在最后一个 epoch 保存 embeddings 和对齐矩阵
        if epoch == 19:
            # 保存每个 epoch 的 embeddings
            save_embeddings_per_patient(torch.cat(all_z_t), torch.cat(all_z_i), args.output_dir, all_patient_ids)

            # 计算并保存对齐矩阵
            alignment_result = calculate_alignment_matrix(torch.cat(all_z_t), torch.cat(all_z_i), all_patient_ids)

            # 输出对齐矩阵为 JSON 格式
            print(json.dumps(alignment_result))

            # 保存模型
            torch.save(net.state_dict(), f"{args.output_dir}/tcmt.pt")

if __name__ == "__main__":
    main()

