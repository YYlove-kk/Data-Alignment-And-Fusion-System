import os
import json
import numpy as np
from neo4j import GraphDatabase

# 连接到 Neo4j 数据库
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "12345678"))

# 加载文本嵌入
def load_text_embeddings(embedding_dir, text_filenames):
    embeddings = {}
    for text_filename in text_filenames:
        embedding_path = os.path.join(embedding_dir, text_filename)
        # 确保文件存在
        if os.path.exists(embedding_path):
            embeddings[text_filename] = np.load(embedding_path)
        else:
            print(f"警告: {embedding_path} 文件不存在。")
    return embeddings

# 计算余弦相似度
def calculate_similarity(embedding1, embedding2):
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
    return similarity

# 创建 Neo4j 图谱
def create_neo4j_graph(embeddings, tag):
    session = driver.session()

    try:
        # 清空数据库（可选）
        # session.run("MATCH (n) DETACH DELETE n")

        # 创建带有 tag 的 Text 节点
        for text_id, embedding in embeddings.items():
            session.run("""
            MERGE (txt:Text {id: $id, tag: $tag})
            """, id=text_id, tag=tag)

        # 计算相似度并创建边
        text_ids = list(embeddings.keys())
        for i, text_id1 in enumerate(text_ids):
            for j, text_id2 in enumerate(text_ids):
                if i >= j:
                    continue
                similarity = calculate_similarity(embeddings[text_id1], embeddings[text_id2])
                if similarity > 0.7:  # 只保留相似度高于阈值的边
                    session.run("""
                    MATCH (a:Text {id: $id1, tag: $tag}), (b:Text {id: $id2, tag: $tag})
                    MERGE (a)-[:SIMILAR {score: $score, tag: $tag}]->(b)
                    """, id1=text_id1, id2=text_id2, score=similarity, tag=tag)

        print("✅ 图谱构建完成。")
    finally:
        session.close()

# 主程序入口
if __name__ == "__main__":
    # 示例：Java 传入的 Set<String>（这里只是示例）
    text_filenames = {"text1.npy", "text2.npy", "text3.npy"}  # 示例文件名
    tag = "example_tag"  # 用于给节点和边打标签

    embedding_dir = "data/upload/output/text"
    embeddings = load_text_embeddings(embedding_dir, text_filenames)
    create_neo4j_graph(embeddings, tag)
