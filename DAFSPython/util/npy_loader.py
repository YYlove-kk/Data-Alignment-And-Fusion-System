import os
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch

class EmbeddingDataset(Dataset):
    def __init__(self, base_dir, split='train', test_size=0.2, device=None):
        self.base_dir = base_dir
        self.split = split  # 'train' 或 'test'
        self.device = device
        self.pair_list = self._load_data(test_size)

    def _load_data(self, test_size):
        txt_dir = os.path.join(self.base_dir, 'txt')
        img_dir = os.path.join(self.base_dir, 'img')

        if not os.path.exists(txt_dir) or not os.path.exists(img_dir):
            raise FileNotFoundError(f"Cannot find txt or img subdirectory in {self.base_dir}")

        txt_files = [f for f in os.listdir(txt_dir) if f.endswith('_z_t.npy')]
        img_files = set(os.listdir(img_dir))

        pair_list = []
        for txt_file in txt_files:
            patient_id = txt_file.replace('_z_t.npy', '')
            img_file = f"{patient_id}_z_i.npy"
            if img_file in img_files:
                txt_path = os.path.join(txt_dir, txt_file)
                img_path = os.path.join(img_dir, img_file)
                pair_list.append((txt_path, img_path))

        if not pair_list:
            raise ValueError(f"No matched vector pairs found in {self.base_dir}/txt and /img.")

        train_pairs, test_pairs = train_test_split(pair_list, test_size=test_size, random_state=42)

        if self.split == 'train':
            return train_pairs
        else:
            return test_pairs

    def __len__(self):
        return len(self.pair_list)

    def __getitem__(self, idx):
        text_file, image_file = self.pair_list[idx]
        text_vec = np.load(text_file)
        image_vec = np.load(image_file)

        if text_vec.ndim == 2:
            text_vec = text_vec.squeeze(0)
        if image_vec.ndim == 2:
            image_vec = image_vec.squeeze(0)

        # 随机决定是否构造负样本
        if np.random.rand() < 0.7:
            label = 1
        else:
            neg_idx = (idx + np.random.randint(1, len(self.pair_list))) % len(self.pair_list)
            _, neg_image_file = self.pair_list[neg_idx]
            image_vec = np.load(neg_image_file)
            if image_vec.ndim == 2:
                image_vec = image_vec.squeeze(0)
            label = 0

        # 转成 Tensor
        text_vec = torch.tensor(text_vec, dtype=torch.float32).to(self.device)
        image_vec = torch.tensor(image_vec, dtype=torch.float32).to(self.device)
        label = torch.tensor([label], dtype=torch.float32).to(self.device)

        return text_vec, image_vec, label
