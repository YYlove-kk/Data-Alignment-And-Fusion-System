# alignment_utils.py
import datetime
import glob
import os


import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class PairDataset(Dataset):
    def __init__(self, match_index, text_dir, image_dir, split=1, train=True):
        self.match_index = match_index
        self.text_dir = text_dir
        self.image_dir = image_dir

    def __len__(self):
        return len(self.match_index)

    def __getitem__(self, idx):
        entry = self.match_index[idx]
        patient_id = entry["patient_id"]
        text_idx = entry["text_idx"]
        image_idx = entry["image_idx"]

        t_vec = np.load(os.path.join(self.text_dir, f"{patient_id}.npy"))[text_idx]
        i_vec = np.load(os.path.join(self.image_dir, f"{patient_id}.npy"))[image_idx]

        return torch.tensor(t_vec, dtype=torch.float32), torch.tensor(i_vec, dtype=torch.float32), patient_id


def extract_date_from_filename(name):
    for part in name.split("_"):
        if part.isdigit() and len(part) == 14:
            return datetime.strptime(part, "%Y%m%d%H%M%S")
    return None

def extract_patient_id_from_filename(name):
    parts = name.split("_")
    if len(parts) > 1:
        return parts[0]
    return "unknown"

def load_npy_with_date(npy_dir):
    data = []
    for path in glob.glob(os.path.join(npy_dir, "*.npy")):
        date = extract_date_from_filename(os.path.basename(path))
        if date:
            data.append({
                "file": os.path.basename(path),
                "date": date
            })
    return data

def match_by_date(text_data, image_data):
    matches = []
    used_images = set()
    for text_item in text_data:
        best_match = None
        best_time_diff = datetime.timedelta.max
        for image_item in image_data:
            if image_item['file'] in used_images:
                continue
            time_diff = abs(text_item['date'] - image_item['date'])
            if time_diff < best_time_diff:
                best_match = image_item
                best_time_diff = time_diff
        if best_match:
            used_images.add(best_match['file'])
            matches.append((text_item['file'], best_match['file']))
    return matches

def get_patient_ids_and_save(text_dir, image_dir, match_dir):
    os.makedirs(match_dir, exist_ok=True)

    print("加载嵌入...")
    text_data = load_npy_with_date(text_dir)
    image_data = load_npy_with_date(image_dir)

    print(f"表格数: {len(text_data)}, 影像数: {len(image_data)}")
    print("匹配中...")

    matches = match_by_date(text_data, image_data)
    print(f"匹配对数: {len(matches)}")

    index = []

    for text_file, image_file in matches:
        patient_id = extract_patient_id_from_filename(image_file)
        index.append({
            "patient_id": patient_id,
            "text_file": text_file,
            "image_file": image_file
        })

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    match_index_file = os.path.join(match_dir, f"match_index_{timestamp}.csv")
    pd.DataFrame(index).to_csv(match_index_file, index=False)
    print(f"✅ 匹配完成，保存至 {match_index_file}")

    return index

