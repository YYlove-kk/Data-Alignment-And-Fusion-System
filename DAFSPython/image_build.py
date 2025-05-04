import os
import json
import numpy as np
from neo4j import GraphDatabase

# 连接到 Neo4j 数据库
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "12345678"))

def load_image_embeddings(embedding_dir, image_filenames):
    embeddings = {}
    for image_filename in image_filenames:
        embedding_path = os.path.join(embedding_dir, image_filename)
        # 确保文件存在
        if os.path.exists(embedding_path):
            embeddings[image_filename] = np.load(embedding_path)
        else:
            print(f"警告: {embedding_path} 文件不存在。")
    return embeddings

def calculate_similarity(embedding1, embedding2):
    # 计算余弦相似度
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
    return similarity

def create_neo4j_graph(embeddings, tag):
    session = driver.session()

    try:

        # 创建带有 tag 的 Image 节点
        for image_id, embedding in embeddings.items():
            session.run("""
            MERGE (img:Image {id: $id, tag: $tag})
            """, id=image_id, tag=tag)

        image_ids = list(embeddings.keys())
        for i, image_id1 in enumerate(image_ids):
            for j, image_id2 in enumerate(image_ids):
                if i >= j:
                    continue
                similarity = calculate_similarity(embeddings[image_id1], embeddings[image_id2])
                if similarity > 0.7:  # 只保留相似度高于阈值的边
                    session.run("""
                    MATCH (a:Image {id: $id1, tag: $tag}), (b:Image {id: $id2, tag: $tag})
                    MERGE (a)-[:SIMILAR {score: $score}]->(b)
                    """, id1=image_id1, id2=image_id2, score=similarity, tag=tag)

        print("✅ 图谱构建完成。")
    finally:
        session.close()

if __name__ == "__main__":
    # 示例：Java 传入的包含文件名的 Set<String>（这里只是示例）
    image_filenames = {"image1.npy", "image2.npy", "image3.npy"}  # 示例文件名
    tag = "example_tag"  # 用于给节点和边打标签

    embedding_dir = "data/upload/output/image"
    embeddings = load_image_embeddings(embedding_dir, image_filenames)
    create_neo4j_graph(embeddings, tag)
