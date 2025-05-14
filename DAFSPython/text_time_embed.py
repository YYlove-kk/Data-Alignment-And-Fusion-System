import argparse
import json
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from util.time2vec import Time2Vec

PROMPT_MAP = {
    "patient_id": "患者ID为：",
    "age": "年龄为：",
    "gender": "性别为：",
    "visit_time": "就诊时间为：",
    "name": "姓名为：",
    "admission_time": "入院时间为：",
    "discharge_time": "出院时间为：",
    "chief_complaint": "主诉为：",
    "history": "病史为：",
    "disease_name": "诊断为："
}

def safe_to_datetime(date_str):
    try:
        return pd.to_datetime(date_str)
    except Exception:
        return pd.NaT  # 对于无效日期返回 NaT

def line_to_prompt(row: pd.Series) -> str:
    """将一行转为 prompt 字符串"""
    parts = []
    for col, val in row.items():
        if pd.isna(val):
            continue
        tag = PROMPT_MAP.get(col.lower(), f"<{col.upper()}>")
        parts.append(f"{tag} {val}")
    return " ; ".join(parts)

def get_bert_embeddings(texts, tokenizer, model, device='cpu', batch_size=32, max_length=128):
    embeddings = []
    model.eval()
    for i in tqdm(range(0, len(texts), batch_size), desc="Text Embedding..."):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            cls_embeds = outputs.last_hidden_state[:, 0, :]  # 取 [CLS]

        # 检查是否有 NaN
        if np.any(np.isnan(cls_embeds.cpu().numpy())):
            print(f"[WARN] 检测到 NaN 在文本嵌入的第 {i} 批次中")

        embeddings.append(cls_embeds.cpu().numpy())

    return np.vstack(embeddings)

def embed_dataframe(pq_file: str, out_dir: str, id: str) -> list:
    df = pd.read_parquet(pq_file)
    print("Parquet 实际列名：", df.columns.tolist())

    df['__row_index'] = np.arange(len(df))
    df = df.sort_values(by=['visit_time', '__row_index']).reset_index(drop=True)

    # 合并重复的 name 和 visit_time
    df_grouped = df.groupby(['name', 'visit_time'], as_index=False).agg({
        'gender': 'first',
        'age': 'first',
        'VISIT_DEPT_NAME': 'first',
        'DISCHARGE_DEPT_NAME': 'first',
        'discharge_time': 'first',
        'VISIT_ORD_NO': 'first',
        'INPATIENT_ORD_NO': 'first',
        'IMAGING_FINDING': ' '.join,
        '项目名': ' '.join,
        'chief_complaint': ' '.join,
        'history': ' '.join,
        'disease_name': ' '.join,
    })

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
    bert = BertModel.from_pretrained('hfl/chinese-roberta-wwm-ext').to(device)
    time2vec = Time2Vec(dim=32)

    # ---------- 保留合法行 ----------
    valid_rows = []
    valid_prompts = []
    valid_times = []

    for i, row in df_grouped.iterrows():
        ts = safe_to_datetime(row["visit_time"])
        if pd.isna(ts):
            print(f"[WARN] 跳过无效 visit_time 行 index={i}")
            continue
        prompt = line_to_prompt(row)
        valid_rows.append((i, row))
        valid_prompts.append(prompt)
        valid_times.append(ts)

    if not valid_rows:
        raise ValueError("未找到任何合法的行，无法生成嵌入")

    # ---------- 嵌入生成 ----------
    print(f"有效样本数：{len(valid_prompts)}")
    text_embeds = get_bert_embeddings(valid_prompts, tokenizer, bert, device=device)

    # 检查 text_embeds 是否有 NaN
    if np.any(np.isnan(text_embeds)):
        print("[ERROR] 检测到 text_embeds 中有 NaN")

    time_embeds = np.vstack([time2vec(ts).detach().numpy() for ts in valid_times])

    # 检查 time_embeds 是否有 NaN
    if np.any(np.isnan(time_embeds)):
        print("[ERROR] 检测到 time_embeds 中有 NaN")

    final_embeds = np.concatenate([text_embeds, time_embeds], axis=1)

    # ---------- NAN 检查 ----------
    nan_rows = np.any(np.isnan(final_embeds), axis=1)
    if np.any(nan_rows):
        print(f"[ERROR] 检测到 {np.sum(nan_rows)} 行包含 NaN，对应索引如下：")
        print(np.where(nan_rows)[0])

    # ---------- 保存 ----------
    os.makedirs(out_dir, exist_ok=True)
    saved_paths = []

    for idx, (original_i, row) in enumerate(valid_rows):
        if np.any(np.isnan(final_embeds[idx])) or np.all(final_embeds[idx] == 0):
            print(f"[SKIP] 跳过无效嵌入：{id}_{idx}")
            continue

        visit_time = safe_to_datetime(row["visit_time"]).strftime("%Y%m%d") if not pd.isna(row["visit_time"]) else "unknown_time"
        filename = f"{id}_{visit_time}_{idx}_txt.npy"
        output_path = os.path.join(out_dir, filename)
        np.save(output_path, final_embeds[idx])
        saved_paths.append(filename)

    print(f"[DONE] 成功保存 {len(saved_paths)} 个嵌入向量")
    return saved_paths


def main():

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--file_path", required=True)
    # parser.add_argument("--output_dir", required=True)
    # args = parser.parse_args()
    #
    # txt_id = os.path.splitext(os.path.basename(args.file_path))[0]
    # paths = embed_dataframe(args.file_path, args.output_dir, txt_id)

    file_path = "../data/upload/clean/test.parquet"
    output_dir = "../data/align/raw/text"

    txt_id = os.path.splitext(os.path.basename(file_path))[0]
    paths = embed_dataframe(file_path, output_dir, txt_id)

    # ✅ 输出 JSON 到标准输出
    print(json.dumps({"paths": paths}))

if __name__ == "__main__":
    main()
