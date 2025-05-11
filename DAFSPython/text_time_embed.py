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

        embeddings.append(cls_embeds.cpu().numpy())

    return np.vstack(embeddings)

def embed_dataframe(pq_file: str, out_dir: str, id: str) -> list:
    df = pd.read_parquet(pq_file)
    print("Parquet 实际列名：", df.columns.tolist())

    df['__row_index'] = np.arange(len(df))
    df = df.sort_values(by=['visit_time', '__row_index']).reset_index(drop=True)

    # 合并重复的 name 和 visit_time
    df_grouped = df.groupby(['name', 'visit_time'], as_index=False).agg({
        'gender': 'first',  # 假设每个患者的性别是相同的
        'age': 'first',          # 假设年龄不变
        'VISIT_DEPT_NAME': 'first',
        'DISCHARGE_DEPT_NAME': 'first',
        'discharge_time': 'first',  # 使用 discharge_time 代替 DISCHARGE_DATE
        'VISIT_ORD_NO': 'first',  # 假设每次就诊编号唯一
        'INPATIENT_ORD_NO': 'first',
        'IMAGING_FINDING': ' '.join,  # 将影像发现拼接
        '项目名': ' '.join,  # 将项目名拼接
        'chief_complaint': ' '.join,  # 主诉拼接
        'history': ' '.join,  # 病史拼接
        'disease_name': ' '.join,  # 诊断拼接
    })



    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
    bert = BertModel.from_pretrained('hfl/chinese-roberta-wwm-ext').to(device)

    time2vec = Time2Vec(dim=32)

    prompts = df_grouped.apply(line_to_prompt, axis=1).tolist()
    text_embeds = get_bert_embeddings(prompts, tokenizer, bert, device=device)

    time_embeds = []
    for date_str in tqdm(df_grouped["visit_time"], desc="Time Embedding..."):
        ts = safe_to_datetime(date_str)  # 使用 safe_to_datetime 进行转换
        if pd.isna(ts):  # 如果时间转换失败，则跳过
            continue
        vec = time2vec(ts).detach().numpy()
        time_embeds.append(vec)
    if not time_embeds:
        raise ValueError("没有有效的时间嵌入数据，无法进行堆叠")
    time_embeds = np.stack(time_embeds)


    final_embeds = np.concatenate([text_embeds, time_embeds], axis=1)

    os.makedirs(out_dir, exist_ok=True)
    saved_paths = []

    for i, row in df_grouped.iterrows():
        visit_time = safe_to_datetime(row["visit_time"]).strftime("%Y%m%d") if not pd.isna(row["visit_time"]) else "unknown_time"
        filename = f"{id}_{visit_time}_{i}_txt.npy"
        output_path = os.path.join(out_dir, filename)
        np.save(output_path, final_embeds[i])
        saved_paths.append(filename)

    return saved_paths

def main():

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--file_path", required=True)
    # parser.add_argument("--output_dir", required=True)
    # args = parser.parse_args()
    #
    # txt_id = os.path.splitext(os.path.basename(args.file_path))[0]
    # paths = embed_dataframe(args.file_path, args.output_dir, txt_id)

    file_path = "../data/upload/clean/test例.parquet"
    output_dir = "../data/align/raw/text"

    txt_id = os.path.splitext(os.path.basename(file_path))[0]
    paths = embed_dataframe(file_path, output_dir, txt_id)

    # ✅ 输出 JSON 到标准输出
    print(json.dumps({"paths": paths}))

if __name__ == "__main__":
    main()
