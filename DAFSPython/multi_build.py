import os
import base64
import sys
import zlib
import numpy as np
import pymysql
from neo4j import GraphDatabase
import hnswlib
from scipy.spatial.distance import cosine


# 连接到 Neo4j 数据库
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "12345678"))

mysql_config = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': '123456',
    'database': 'dafs',
    'charset': 'utf8mb4',
}

connection = pymysql.connect(**mysql_config)
cursor = connection.cursor()

def load_joint_relations_from_mysql():
    sql = "SELECT patient_id, text_file, image_file FROM joint_embedding_relation"
    cursor.execute(sql)
    rows = cursor.fetchall()

    relations = {}
    for patient_id, text_file, image_file in rows:
        relations[patient_id] = (text_file, image_file)

    return relations

# 压缩向量为 base64 字符串，便于使用 APOC 存储
def compress_vector(vector):
    """将 numpy 向量压缩为 base64 字符串以便用 APOC 存储"""
    header = np.array([vector.shape[0]], dtype=np.int32).tobytes()
    compressed = zlib.compress(header + vector.tobytes())
    return base64.b64encode(compressed).decode('utf-8')

# 向 Neo4j 写入 Visit 节点
def write_visit_nodes(tx, patient_id, hash_id, vector_b64, tag):
    # 先确保 Patient 节点存在，并且包含患者的名称属性
    cypher_patient = """
    MERGE (p:Patient {id: $patient_id, tag: $tag})
    """
    tx.run(cypher_patient, patient_id=patient_id, tag=tag)

    # 写入 Visit 节点，加入 tag 属性
    cypher_visit = """
    MERGE (v:Visit {hash: $hash, tag: $tag})
    SET v.embedding = $vector
    MATCH (p:Patient {id: $patient_id, tag: $tag})
    MERGE (v)-[:BELONGS_TO]->(p)
    SET v.embedding = $vector
    """
    tx.run(cypher_visit, patient_id=patient_id, hash=hash_id, vector=vector_b64, tag=tag)

# 写入 Text/Image 节点，并连接 Visit 节点
def write_text_image_nodes(tx, visit_hash, text_file, image_file, tag):
    # Text 节点
    cypher_text = """
    MERGE (t:Text {file: $text_file, tag: $tag})
    """
    tx.run(cypher_text, text_file=text_file, tag=tag)

    # Image 节点
    cypher_image = """
    MERGE (i:Image {file: $image_file, tag: $tag})
    """
    tx.run(cypher_image, image_file=image_file, tag=tag)

    # 建立 Visit -> Text 关系
    cypher_link_text = """
    MATCH (v:Visit {hash: $hash, tag: $tag})
    MATCH (t:Text {file: $text_file, tag: $tag})
    MERGE (v)-[:FUSED_FROM]->(t)
    """
    tx.run(cypher_link_text, hash=visit_hash, text_file=text_file, tag=tag)

    # 建立 Visit -> Image 关系
    cypher_link_image = """
    MATCH (v:Visit {hash: $hash, tag: $tag})
    MATCH (i:Image {file: $image_file, tag: $tag})
    MERGE (v)-[:FUSED_FROM]->(i)
    """
    tx.run(cypher_link_image, hash=visit_hash, image_file=image_file, tag=tag)



# 向 Neo4j 创建相似度关系
def create_similarity_edge(tx, src_hash, tgt_hash, weight, tag):
    cypher = """
    MATCH (v1:Visit {hash: $src, tag: $tag})
    MATCH (v2:Visit {hash: $tgt, tag: $tag})
    MERGE (v1)-[r:SIMILAR]->(v2)
    SET r.weight = $weight, r.tag = $tag
    """
    tx.run(cypher, src=src_hash, tgt=tgt_hash, weight=weight, tag=tag)


def create_visit_nodes_from_aligned_vectors(aligned_dir, patient_ids, tag, relations):
    all_visit_ids = []  # 存储所有 Visit 节点 ID
    all_vectors = []  # 存储所有向量数据

    for patient_id in patient_ids:
        for filename in os.listdir(aligned_dir):
            if filename.endswith(patient_id + ".npy"):  # 仅处理当前患者的文件
                path = os.path.join(aligned_dir, filename)
                aligned_vectors = np.load(path)  # 加载对齐后的向量
                for idx, vec in enumerate(aligned_vectors):
                    visit_hash = f"{filename}_{idx}"
                    vec_b64 = compress_vector(vec)  # 压缩为 base64 字符串
                    all_visit_ids.append(visit_hash)
                    all_vectors.append(vec)
                    with driver.session() as session:
                        # 写入 visit 节点，结合 tag
                        session.execute_write(write_visit_nodes, patient_id, visit_hash, vec_b64, tag)
                        # 新增：写入 Text 和 Image 节点及关系
                        if patient_id in relations:
                            text_file, image_file = relations[patient_id]
                            session.execute_write(write_text_image_nodes, visit_hash, text_file, image_file, tag)

                        print(f"Processed aligned Visit node: {visit_hash} for patient {patient_id}")

    return all_visit_ids, all_vectors


# 计算并创建相似度边
def build_similarity_edges_from_aligned_vectors(all_visit_ids, all_vectors, tag):
    vectors_np = np.vstack(all_vectors).astype("float32")
    dim = vectors_np.shape[1]
    num_elements = len(vectors_np)

    # 初始化 hnswlib 索引
    p = hnswlib.Index(space='cosine', dim=dim)
    p.init_index(max_elements=num_elements, ef_construction=200, M=32)
    p.add_items(vectors_np)
    p.set_ef(50)

    labels, distances = p.knn_query(vectors_np, k=10)

    for i, neighbors in enumerate(labels):
        for j, neighbor_idx in enumerate(neighbors):
            if i == neighbor_idx:
                continue
            sim = 1 - cosine(vectors_np[i], vectors_np[neighbor_idx])
            if sim >= 0.7:
                source = all_visit_ids[i]
                target = all_visit_ids[neighbor_idx]
                with driver.session() as session:
                    session.execute_write(create_similarity_edge, source, target, sim, tag)
                    print(f"Created similarity edge: {source} -> {target}")

# 主程序：加载降维后的向量并创建知识图谱
def main():
    if len(sys.argv) != 3:
        print("Usage: python builder.py <patientIds> <tag>")
        sys.exit(1)

    patient_ids = sys.argv[1].split(',')  # 获取多个患者ID
    tag = int(sys.argv[2])

    aligned_dir = "data/align/joint"  # 设置为对齐结果保存的目录

    # 加载文本-影像关系
    relations = load_joint_relations_from_mysql()

    all_visit_ids, all_vectors = create_visit_nodes_from_aligned_vectors(aligned_dir, patient_ids, tag, relations)

    build_similarity_edges_from_aligned_vectors(all_visit_ids, all_vectors, tag)

    print("跨模态知识图谱构建完成！")


if __name__ == "__main__":
    main()