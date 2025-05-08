import os
import glob
import json
import torch
import numpy as np
import argparse
from datetime import datetime
from model.model_tcmt import TCMT
from sklearn.metrics.pairwise import cosine_similarity


def extract_date_from_filename(name):
    for part in name.split("_"):
        if part.isdigit() and len(part) == 14:
            return datetime.strptime(part, "%Y%m%d%H%M%S")
    return None

def extract_patient_id_from_filename(name):
    return name.split("_")[0]

def load_npy_with_date(npy_dir):
    data = []
    for path in glob.glob(os.path.join(npy_dir, "*.npy")):
        date = extract_date_from_filename(os.path.basename(path))
        if date:
            data.append((os.path.basename(path), date))
    return data

def match_text_image(text_dir, image_dir):
    text_data = load_npy_with_date(text_dir)
    image_data = load_npy_with_date(image_dir)

    matches = []
    for txt_file, txt_date in text_data:
        min_diff = float("inf")
        best_img_file = None
        for img_file, img_date in image_data:
            diff = abs((txt_date - img_date).days)
            if diff < min_diff:
                min_diff = diff
                best_img_file = img_file
        if best_img_file:
            matches.append((txt_file, best_img_file))
    return matches

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_path", required=True)
    args = parser.parse_args()

    text_dir = "data/align/text"
    image_dir = "data/align/image"
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TCMT().to(device)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    matches = match_text_image(text_dir, image_dir)

    z_t_all, z_i_all, patient_ids = [], [], []

    for txt_file, img_file in matches:
        txt_vec = torch.from_numpy(np.load(os.path.join(text_dir, txt_file))).float().to(device)
        img_vec = torch.from_numpy(np.load(os.path.join(image_dir, img_file))).float().to(device)

        with torch.no_grad():
            zt, zi = model(txt_vec.unsqueeze(0), img_vec.unsqueeze(0))

        patient_id = extract_patient_id_from_filename(txt_file)
        patient_ids.append(patient_id)

        z_t_np = zt.cpu().squeeze(0).numpy()
        z_i_np = zi.cpu().squeeze(0).numpy()

        np.save(f"{args.output_dir}/{patient_id}_z_t.npy", z_t_np)
        np.save(f"{args.output_dir}/{patient_id}_z_i.npy", z_i_np)

        z_t_all.append(z_t_np)
        z_i_all.append(z_i_np)

    # ✅ 指标计算
    z_t_all = np.stack(z_t_all)
    z_i_all = np.stack(z_i_all)

    similarity_matrix = cosine_similarity(z_t_all, z_i_all)
    diagonal = similarity_matrix.diagonal()
    correct_matches = np.argmax(similarity_matrix, axis=1)
    accuracy = np.mean(correct_matches == np.arange(len(z_t_all)))
    coverage = float((similarity_matrix.max(axis=1) > 0.7).mean())

    alignment_result = {
        "alignment_matrix": similarity_matrix.tolist(),
        "semantic_accuracy": float(accuracy),
        "alignment_coverage": float(coverage),
        "diagonal_similarity": diagonal.tolist(),
        "patient_ids": patient_ids
    }

    print(json.dumps(alignment_result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
