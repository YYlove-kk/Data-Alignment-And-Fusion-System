import torch
import numpy as np
import glob
import random
import os
from torch.utils.data import Dataset, DataLoader
from model.model_tcmt import   TCMT, clip_loss



def extract_tcmt_embeddings(model_path, txt_dir, img_dir, output_dir):
    # 加载模型
    model = TCMT()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # 获取所有样本 id
    ids = [os.path.basename(p).replace("_txt.npy", "") for p in glob.glob(f"{txt_dir}/*_txt.npy")]

    with torch.no_grad():
        for pid in ids:
            txt_path = f"{txt_dir}/{pid}_txt.npy"
            img_path = f"{img_dir}/{pid}_img.npy"

            if not os.path.exists(txt_path) or not os.path.exists(img_path):
                continue

            txt = np.load(txt_path)
            img = np.load(img_path)

            txt_tensor = torch.tensor(txt, dtype=torch.float32).unsqueeze(0)  # [1, 800]
            img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # [1, 2080]

            z_t, z_i = model(txt_tensor, img_tensor)  # [1, 512] each

            # 去掉 batch 维度再保存
            np.save(os.path.join(output_dir, f"{pid}_z_t.npy"), z_t.squeeze(0).cpu().numpy())
            np.save(os.path.join(output_dir, f"{pid}_z_i.npy"), z_i.squeeze(0).cpu().numpy())
            print(f"Saved: {pid}_z_t.npy and {pid}_z_i.npy")


if __name__ == "__main__":
    extract_tcmt_embeddings(
        model_path="output/tcmt_data/tcmt_ep19.pt",  # 你训练保存的 pt 文件
        txt_dir="output/text",
        img_dir="output/video",
        output_dir="output/embeddings_512d"
    )
