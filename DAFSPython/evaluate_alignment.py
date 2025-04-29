# ---------------------------------------------
# File: evaluate_alignment.py
# Role: Evaluate alignment accuracy (Top-1, MRR)
# ---------------------------------------------
import os
import numpy as np
import glob
import tqdm
import torch
from model.model_tcmt import TCMT


def load_pairs(txt_dir, img_dir, model, device):
    ids = []
    T, I = [], []
    for p in tqdm.tqdm(glob.glob(f"{txt_dir}/*.npy")):
        pid = os.path.basename(p).split("_")[0]
        img_path = os.path.join(img_dir, f"{pid}_img.npy")
        if not os.path.exists(img_path):
            continue

        txt = np.load(p)
        img = np.load(img_path)

        txt = torch.tensor(txt, dtype=torch.float32).to(device)
        img = torch.tensor(img, dtype=torch.float32).to(device)

        # 送入模型，拿到输出的 z_t 和 z_i
        with torch.no_grad():
            z_t, z_i = model(txt.unsqueeze(0), img.unsqueeze(0))  # 加batch维度 (1, d)
            z_t = z_t.squeeze(0).cpu().numpy()
            z_i = z_i.squeeze(0).cpu().numpy()

        T.append(z_t)
        I.append(z_i)
        ids.append(pid)
    T = np.vstack(T)
    I = np.vstack(I)
    return T, I, ids


def mrr(T, I):
    sims = T @ I.T  # (N, N)
    ranks = (-sims).argsort(axis=1)
    hits = (ranks == np.arange(len(T))[:, None]).argmax(axis=1) + 1
    return np.mean(1 / hits)


def main(txt_dir, img_dir, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TCMT().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    T, I, ids = load_pairs(txt_dir, img_dir, model, device)

    sims = T @ I.T
    top1 = sims.argmax(axis=1) == np.arange(len(T))
    print("Top-1 Accuracy:", top1.mean())

    mrr_score = mrr(T, I)
    print("MRR (Mean Reciprocal Rank):", mrr_score)


if __name__ == "__main__":

    # 从命令行传入目录路径
    main("output/text", "output/video","output/tcmt_data/tcmt_ep19.pt")
