# ---------------------------------------------
# File: text_time_embed.py
# Role: 文本 + 时间嵌入生成
# ---------------------------------------------
import argparse
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from utils.time2vec import Time2Vec

PROMPT_MAP = {
    "age": "<AGE>",
    "diagnosis": "<DIAG>",
    "sex": "<SEX>",
    # 可按需增加其他列名映射
}


def line_to_prompt(row: pd.Series) -> str:
    """把一行转为 prompt 字符串"""
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


def embed_dataframe(pq_file: str, out_dir: str, id: str):
    df = pd.read_parquet(pq_file)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = BertTokenizer.from_pretrained('E:/graduationProject/project/model')
    bert = BertModel.from_pretrained('E:/graduationProject/project/model').to(device)

    time2vec = Time2Vec(dim=32)

    # 准备文本输入
    prompts = df.apply(line_to_prompt, axis=1).tolist()
    text_embeds = get_bert_embeddings(prompts, tokenizer, bert, device=device)  # shape: (N, 768)

    # 时间嵌入处理
    time_embeds = []
    for date_str in tqdm(df["VISIT_DATE"], desc="Time Embedding..."):
        ts = pd.to_datetime(date_str)
        vec = time2vec(ts).detach().numpy()  # shape: (32,)
        time_embeds.append(vec)
    time_embeds = np.stack(time_embeds)  # shape: (N, 32)

    # 合并文本和时间向量
    final_embeds = np.concatenate([text_embeds, time_embeds], axis=1)  # shape: (N, 800)
    output_file = os.path.join(out_dir, id + "_txt.npy")
    np.save(output_file, final_embeds)
    return output_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    txt_id = os.path.splitext(os.path.basename(args.file_path))[0]
    output_path = embed_dataframe(args.file_path, args.output_dir, txt_id)
    print(output_path)

if __name__ == "__main__":
    main()
