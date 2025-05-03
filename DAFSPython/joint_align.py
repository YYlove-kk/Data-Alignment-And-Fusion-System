import os
import glob
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from tqdm import tqdm
import json
from fastdtw import fastdtw
from scipy.spatial.distance import cosine
from model.model_tcmt import TCMT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_date_from_filename(name):
    # 假设 filename 包含日期字段（如20230421），你也可以按你实际格式调整
    for part in name.split("_"):
        if part.isdigit() and len(part) == 8:
            return datetime.strptime(part, "%Y%m%d").date()
    return None

def extract_patient_name_from_filename(name):
    # 假设 filename 格式为 {patient_name}_{timestamp}_img.npy
    parts = name.split("_")
    if len(parts) > 1:
        return parts[0]  # 获取 patient_name 假设为第二部分
    return None

def load_npy_with_date(npy_dir):
    data = []
    for path in glob.glob(os.path.join(npy_dir, "*.npy")):
        arr = np.load(path)
        date = extract_date_from_filename(os.path.basename(path))
        if date:
            data.append((os.path.basename(path), date, arr))
    return data  # List of (filename, date, np_array)

def match_by_date(text_data, image_data):
    matched = []
    for txt_file, txt_date, txt_vec in text_data:
        # 找到与其日期最接近的图像文件
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

def compute_alignment(reference, sequence, output_dir):
    """
    使用 FastDTW 对参考序列和目标序列进行对齐。
    返回路径对、相似度、coverage、diagonal similarity，并保存对齐后的向量。
    """
    ref_vecs = [vec for _, vec in reference]
    seq_vecs = [vec for _, vec in sequence]
    path, _ = fastdtw(seq_vecs, ref_vecs, dist=cosine)

    aligned_pairs = []
    similarities = []
    diagonal_hits = 0

    # 创建用于保存对齐后的向量的文件夹
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    aligned_seq = []  # 用于保存对齐后的向量
    aligned_filenames = []

    for i, j in path:
        sim = 1 - cosine(seq_vecs[i], ref_vecs[j])
        similarities.append(sim)
        aligned_pairs.append((sequence[i][0], reference[j][0]))

        if abs(i - j) <= 1:  # 可调容差
            diagonal_hits += 1

        # 将对齐后的向量保存为 .npy 文件
        aligned_seq.append(seq_vecs[i])  # 添加对齐后的向量

        # 保存对齐后的向量为新的 .npy 文件
        np.save(os.path.join(output_dir, f"{sequence[i][0]}"), seq_vecs[i])

        aligned_filenames.append({
            "text_file": sequence[i][0],
            "image_file": reference[j][0]
        })

    diagonal_similarity = diagonal_hits / len(path)
    semantic_accuracy = np.mean(similarities)
    coverage = len(set(i for i, _ in path)) / len(seq_vecs)

    return {
        "alignment_matrix": aligned_pairs,
        "diagonal_similarity": diagonal_similarity,
        "semantic_accuracy": semantic_accuracy,
        "coverage": coverage,
        "aligned_filenames": aligned_filenames  # 原文件名列表
    }

def main():
    text_dir = "data/align/output/text"
    image_dir = "data/align/output/image"
    joint_dir = "data/align/output/joint"
    os.makedirs(joint_dir, exist_ok=True)

    print("加载嵌入...")
    text_data = load_npy_with_date(text_dir)
    image_data = load_npy_with_date(image_dir)

    print(f"表格数: {len(text_data)}, 影像数: {len(image_data)}")

    print("匹配中...")
    matches = match_by_date(text_data, image_data)
    print(f"匹配对数: {len(matches)}")

    model = TCMT().to(device)
    model.eval()

    index = []

    joint_sequence = []  # 用于存储联合嵌入序列

    for i, (txt_file, img_file, txt_vec, img_vec) in enumerate(tqdm(matches)):

        # 提取患者名称
        patient_name = extract_patient_name_from_filename(img_file)

        txt_tensor = torch.tensor(txt_vec, dtype=torch.float32).unsqueeze(0).unsqueeze(1).to(device)  # (1,1,800)
        img_tensor = torch.tensor(img_vec, dtype=torch.float32).unsqueeze(0).unsqueeze(1).to(device)  # (1,1,2080)

        with torch.no_grad():
            z_t, z_i = model(txt_tensor, img_tensor)  # (1,512), (1,512)

        joint_vec = torch.cat([z_t, z_i], dim=-1).squeeze(0).cpu().numpy()  # (1024,)

        # 使用 patient_name 来命名联合嵌入文件
        joint_name = f"joint_{patient_name}.npy"
        joint_sequence.append((joint_name, joint_vec))  # 直接存入联合嵌入序列

        index.append({
            "joint_id": joint_name,
            "text_file": txt_file,
            "image_file": img_file
        })

        # 使用当前时间戳生成唯一的文件名，避免覆盖
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    joint_index_file = os.path.join(joint_dir, f"joint_index_{timestamp}.csv")
    print(f"[CSV_PATH]{joint_index_file}")  # stdout 直接返回 csv 文件路径


    df = pd.DataFrame(index)
    df.to_csv(joint_index_file, index=False)
    print("联合嵌入完成，保存至 joint_index.csv")

    print("对齐中...")

    # 使用联合嵌入序列进行对齐
    result = compute_alignment(joint_sequence, joint_sequence, joint_dir)

    # 保存对齐结果
    output_json = os.path.join(joint_dir, "alignment_result.json")
    print("保存结果到 JSON...")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"✅ 对齐完成，结果保存在 {output_json}")
    print(f"✅ 对齐后的向量保存在 {joint_dir} 目录下")

if __name__ == "__main__":
    main()
