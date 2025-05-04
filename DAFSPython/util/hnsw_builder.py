import os
import sys
import glob
import hnswlib
import numpy as np
from neo4j import GraphDatabase

# 连接到 Neo4j 数据库
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "12345678"))

# 获取所有 uuid → tag 的映射
def get_node_tag_map(driver):
    tag_map = {}
    with driver.session() as session:
        result = session.run("""
        MATCH (n)
        WHERE exists(n.uuid) AND exists(n.tag)
        RETURN n.uuid AS uuid, n.tag AS tag
        """)
        for record in result:
            tag_map[record["uuid"]] = record["tag"]
    return tag_map

def build_hnsw_index(text_embed_paths, image_embed_paths, dim):
    p = hnswlib.Index(space='cosine', dim=dim * 2)  # 因为要拼接，维度翻倍
    p.init_index(max_elements=len(text_embed_paths), ef_construction=200, M=16)
    index_to_uuid = {}
    for i, (text_path, image_path) in enumerate(zip(text_embed_paths, image_embed_paths)):
        # 直接加载已对齐的嵌入向量
        text_vec = np.load(text_path).astype(float)
        image_vec = np.load(image_path).astype(float)

        # 拼接文本向量和图像向量
        combined_vec = np.concatenate((text_vec, image_vec))

        p.add_items(combined_vec.reshape(1, -1))
        uuid = text_path.split("/")[-1].split(".")[0]
        index_to_uuid[i] = uuid
    return p, index_to_uuid


# 构建跨模态相似边
def create_similar_edges(p, index_to_uuid, driver, tag):
    with driver.session() as s:
        for i in range(len(index_to_uuid)):
            labels, distances = p.knn_query(p.get_items([i])[0], k=20)
            for j, label in enumerate(labels[0]):
                if i != label:
                    uuid1, uuid2 = index_to_uuid[i], index_to_uuid[label]
                    sim = 1 - distances[0][j]
                    if sim >= 0.7:
                        s.run("""
                        MATCH (vt:VisitText {uuid: $uuid1, tag: $tag}), (vi:VisitImage {uuid: $uuid2, tag: $tag})
                        MERGE (vt)-[:MULTI_MODAL_SIMILAR {weight: $weight, tag: $tag}]->(vi)
                        """, uuid1=uuid1, uuid2=uuid2, weight=sim, tag=tag)
                        s.run("""
                        MATCH (vi:VisitImage {uuid: $uuid1, tag: $tag}), (vt:VisitText {uuid: $uuid2, tag: $tag})
                        MERGE (vi)-[:MULTI_MODAL_SIMILAR {weight: $weight, tag: $tag}]->(vt)
                        """, uuid1=uuid1, uuid2=uuid2, weight=sim, tag=tag)

# 构建单模态相似边
def create_single_modal_edges(p, index_to_uuid, driver, tag, is_text):
    node_label = "VisitText" if is_text else "VisitImage"
    with driver.session() as s:
        for i in range(len(index_to_uuid)):
            labels, distances = p.knn_query(p.get_items([i])[0], k=20)
            for j, neighbor_idx in enumerate(labels[0]):
                if i != neighbor_idx:
                    uuid1 = index_to_uuid[i]
                    uuid2 = index_to_uuid[neighbor_idx]
                    sim = 1 - distances[0][j]
                    if sim >= 0.7:
                        s.run(f"""
                        MATCH (a:{node_label} {{uuid: $uuid1, tag: $tag}}), (b:{node_label} {{uuid: $uuid2, tag: $tag}})
                        MERGE (a)-[:SINGLE_MODAL_SIMILAR {{weight: $weight, tag: $tag}}]->(b)
                        """, uuid1=uuid1, uuid2=uuid2, weight=sim, tag=tag)


def run_hnsw_builder(patient_ids, tag, dim, mode):
    try:
        text_embed_paths = []
        image_embed_paths = []

        for patient_id in patient_ids:
            text_path = f"data/align/reduce/{patient_id}_z_t.npy"
            image_path = f"data/align/reduce/{patient_id}_z_i.npy"

            if os.path.exists(text_path):
                text_embed_paths.append(text_path)
            else:
                print(f"未找到文本嵌入文件：{text_path}")

            if os.path.exists(image_path):
                image_embed_paths.append(image_path)
            else:
                print(f"未找到图像嵌入文件：{image_path}")

        if mode == 0:
            print("执行跨模态图谱构建...")
            p, index_to_uuid = build_hnsw_index(text_embed_paths, image_embed_paths, dim)
            create_similar_edges(p, index_to_uuid, driver, tag)

        elif mode == 1:
            print("执行单模态图谱构建...")

            # 文本索引
            p_text = hnswlib.Index(space='cosine', dim=dim)
            p_text.init_index(max_elements=len(text_embed_paths), ef_construction=200, M=16)
            text_index_map = {}
            for i, path in enumerate(text_embed_paths):
                vec = np.load(path).astype(float)
                p_text.add_items(vec.reshape(1, -1))
                uuid = path.split("/")[-1].split(".")[0]
                text_index_map[i] = uuid
            create_single_modal_edges(p_text, text_index_map, driver, tag, is_text=True)

            # 图像索引
            p_image = hnswlib.Index(space='cosine', dim=dim)
            p_image.init_index(max_elements=len(image_embed_paths), ef_construction=200, M=16)
            image_index_map = {}
            for i, path in enumerate(image_embed_paths):
                vec = np.load(path).astype(float)
                p_image.add_items(vec.reshape(1, -1))
                uuid = path.split("/")[-1].split(".")[0]
                image_index_map[i] = uuid
            create_single_modal_edges(p_image, image_index_map, driver, tag, is_text=False)

    except Exception as e:
        print(f"run_hnsw_builder 运行失败 ❌: {e}")

def main():
    if len(sys.argv) != 4:
        print("Usage: python hnsw_builder.py <patientId> <tag> <mode>")
        sys.exit(1)

    patient_ids = sys.argv[1].split(',')
    tag = int(sys.argv[2])
    dim = 256
    mode = int(sys.argv[3])

    run_hnsw_builder(patient_ids, tag, dim, mode)

if __name__ == "__main__":
    main()