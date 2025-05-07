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
        pair_list = []
        # 遍历文件夹中的所有 .npy 文件
        for filename in os.listdir(self.base_dir):
            if filename.endswith("_z_t.npy"):
                patient_id = filename.split("_")[0]  # 提取 patientId
                text_file = os.path.join(self.base_dir, f"{patient_id}_z_t.npy")
                image_file = os.path.join(self.base_dir, f"{patient_id}_z_i.npy")

                # 检查是否存在对应的图像和文本文件
                if os.path.exists(image_file):
                    pair_list.append((text_file, image_file))

        # 划分数据集
        train_pairs, test_pairs = train_test_split(pair_list, test_size=test_size, random_state=42)

        # 根据当前 split 设置数据集
        if self.split == 'train':
            return train_pairs
        elif self.split == 'test':
            return test_pairs

    def __len__(self):
        return len(self.pair_list)

    def __getitem__(self, idx):
        text_file, image_file = self.pair_list[idx]
        text_vec = np.load(text_file)
        image_vec = np.load(image_file)

        # 随机决定是否构造负样本
        if np.random.rand() < 0.5:
            label = 1  # 正样本
        else:
            # 构造负样本：从其他 image 中随机选一个
            neg_idx = (idx + np.random.randint(1, len(self.pair_list))) % len(self.pair_list)
            _, neg_image_file = self.pair_list[neg_idx]
            image_vec = np.load(neg_image_file)
            label = 0

        # 转换为 PyTorch tensor 并移动到指定设备
        text_vec = torch.tensor(text_vec, dtype=torch.float32).to(self.device)
        image_vec = torch.tensor(image_vec, dtype=torch.float32).to(self.device)
        label = torch.tensor([label], dtype=torch.float32).to(self.device)


        return text_vec, image_vec, label
