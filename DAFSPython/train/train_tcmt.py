# train_tcmt.py
import argparse
import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.model_tcmt import TCMT, clip_loss
from util.alignment_utils import  PairDataset, match_by_date


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    text_dir = "data/align/text"
    image_dir = "data/align/image"
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    match_index = match_by_date(text_dir, image_dir)

    ds = PairDataset(match_index, text_dir, image_dir)
    dl = DataLoader(ds, batch_size=64, shuffle=True)

    net = TCMT().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4)

    for epoch in range(20):
        net.train()
        for txt, img, _ in tqdm(dl, desc=f"Epoch {epoch}"):
            txt, img = txt.to(device), img.to(device)
            zt, zi = net(txt, img)
            loss = clip_loss(zt, zi, net.logit_scale.exp())
            loss.backward()
            opt.step()
            opt.zero_grad()

        if epoch == 19:
            torch.save(net.state_dict(), os.path.join(args.output_dir, "tcmt.pt"))
            print("✅ 模型已保存")

if __name__ == "__main__":
    main()
