# match.py
import os
import glob
import pandas as pd
from datetime import datetime
from shutil import copyfile

def extract_date_from_filename(name):
    for part in name.split("_"):
        if part.isdigit() and len(part) == 8:
            return datetime.strptime(part, "%Y%m%d")
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

    pairs = []
    for txt_file, txt_date in text_data:
        for img_file, img_date in image_data:
            diff = abs((txt_date - img_date).days)
            pairs.append((diff, txt_file, img_file))

    # 按时间差升序排序
    pairs.sort(key=lambda x: x[0])

    used_txt = set()
    used_img = set()
    matches = []

    for diff, txt_file, img_file in pairs:
        if txt_file not in used_txt and img_file not in used_img:
            matches.append((txt_file, img_file))
            used_txt.add(txt_file)
            used_img.add(img_file)

    return matches

def main():
    text_dir = "../data/align/reduced/text"
    image_dir = "../data/align/reduced/image"
    out_txt_dir = "../data/align/match/txt"
    out_img_dir = "../data/align/match/img"
    os.makedirs(out_txt_dir, exist_ok=True)
    os.makedirs(out_img_dir, exist_ok=True)

    matches = match_text_image(text_dir, image_dir)
    records = []

    for idx, (txt_file, img_file) in enumerate(matches):
        # 从 image 文件名中提取 patient_id
        patient_id = extract_patient_id_from_filename(img_file)

        # new_txt = f"{patient_id}_txt.npy"
        # new_img = f"{patient_id}_img.npy"

        new_txt = f"{patient_id}_z_t.npy"
        new_img = f"{patient_id}_z_i.npy"

        copyfile(os.path.join(text_dir, txt_file), os.path.join(out_txt_dir, new_txt))
        copyfile(os.path.join(image_dir, img_file), os.path.join(out_img_dir, new_img))

        records.append({
            "patient_id": patient_id,
            "text_file": txt_file,
            "image_file": img_file
        })

    timestamp = datetime.now().strftime("%Y%m%d")
    os.makedirs("../data/align/matchCSV", exist_ok=True)
    match_index_file = os.path.join("../data/align/matchCSV", f"match_index_{timestamp}.csv")
    pd.DataFrame(records).to_csv(match_index_file, index=False)
    print(f"CSV_PATH {match_index_file}")


if __name__ == "__main__":
    main()