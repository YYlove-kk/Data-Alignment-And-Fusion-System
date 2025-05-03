import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from neo4j import GraphDatabase

# 连接到 Neo4j 数据库
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

# 读取 joint_index.csv 文件
joint_index = pd.read_csv("joint_index.csv")

# 计算余弦相似度
def compute_cosine_similarity(vector1, vector2):
    return cosine_similarity([vector1], [vector2])[0][0]

# 创建文本之间的相似度关系
def create_text_similarity_relation(tx, text_joint_id1, text_joint_id2, similarity_score):
    if similarity_score > 0.7:  # 设置相似度阈值为 0.7
        cypher = """
        MATCH (t1:Text {joint_id: $text_joint_id1}), (t2:Text {joint_id: $text_joint_id2})
        MERGE (t1)-[:SIMILAR]->(t2)
        """
        tx.run(cypher, text_joint_id1=text_joint_id1, text_joint_id2=text_joint_id2)

# 创建图像之间的相似度关系
def create_image_similarity_relation(tx, image_joint_id1, image_joint_id2, similarity_score):
    if similarity_score > 0.7:  # 设置相似度阈值为 0.7
        cypher = """
        MATCH (i1:Image {joint_id: $image_joint_id1}), (i2:Image {joint_id: $image_joint_id2})
        MERGE (i1)-[:SIMILAR]->(i2)
        """
        tx.run(cypher, image_joint_id1=image_joint_id1, image_joint_id2=image_joint_id2)

# 遍历 joint_index.csv 文件进行相似度计算
def build_single_modality_graph():
    with driver.session() as session:
        # 加载文本和图像嵌入
        text_vectors = {}
        image_vectors = {}

        for _, row in joint_index.iterrows():
            text_file = row['text_file']
            image_file = row['image_file']
            text_vector = np.load(f"data/upload/output/text/{text_file}")
            image_vector = np.load(f"data/upload/output/image/{image_file}")

            text_vectors[row['joint_id']] = text_vector
            image_vectors[row['joint_id']] = image_vector

        # 创建文本之间的相似度关系
        for text_id1, text_vector1 in text_vectors.items():
            for text_id2, text_vector2 in text_vectors.items():
                if text_id1 != text_id2:
                    similarity_score = compute_cosine_similarity(text_vector1, text_vector2)
                    session.execute_write(create_text_similarity_relation, text_id1, text_id2, similarity_score)

        # 创建图像之间的相似度关系
        for image_id1, image_vector1 in image_vectors.items():
            for image_id2, image_vector2 in image_vectors.items():
                if image_id1 != image_id2:
                    similarity_score = compute_cosine_similarity(image_vector1, image_vector2)
                    session.execute_write(create_image_similarity_relation, image_id1, image_id2, similarity_score)

print("单模态知识图谱构建完成")
