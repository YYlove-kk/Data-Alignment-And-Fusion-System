# ---------------------------------------------
# File: text_time_embed.py
# Role: 文本 + 时间嵌入生成（按行输出）
# ---------------------------------------------
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

        embeddings.append(cls_embeds.cpu().numpy())

    return np.vstack(embeddings)

def embed_dataframe(pq_file: str, out_dir: str, id: str) -> list:
    df = pd.read_parquet(pq_file)
    df['__row_index'] = np.arange(len(df))
    df = df.sort_values(by=['visit_time', '__row_index']).reset_index(drop=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = BertTokenizer.from_pretrained('model')
    bert = BertModel.from_pretrained('model').to(device)

    time2vec = Time2Vec(dim=32)

    prompts = df.apply(line_to_prompt, axis=1).tolist()
    text_embeds = get_bert_embeddings(prompts, tokenizer, bert, device=device)

    time_embeds = []
    for date_str in tqdm(df["visit_time"], desc="Time Embedding..."):
        ts = pd.to_datetime(date_str)
        vec = time2vec(ts).detach().numpy()
        time_embeds.append(vec)
    time_embeds = np.stack(time_embeds)

    final_embeds = np.concatenate([text_embeds, time_embeds], axis=1)

    os.makedirs(out_dir, exist_ok=True)
    saved_paths = []

    for i, row in df.iterrows():
        visit_time = pd.to_datetime(row["visit_time"]).strftime("%Y%m%d%H%M%S")
        filename = f"{id}_{visit_time}_{i}_txt.npy"
        output_path = os.path.join(out_dir, filename)
        np.save(output_path, final_embeds[i])
        saved_paths.append(filename)

    return saved_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    txt_id = os.path.splitext(os.path.basename(args.file_path))[0]
    paths = embed_dataframe(args.file_path, args.output_dir, txt_id)

    # ✅ 输出 JSON 到标准输出
    print(json.dumps({"paths": paths}))

if __name__ == "__main__":
    main()
