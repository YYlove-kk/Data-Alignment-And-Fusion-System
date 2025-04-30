import sys

import numpy as np
from neo4j import GraphDatabase
import glob

# 连接到 Neo4j 数据库
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "12345678"))

def import_patient(pid, text_embed_path, image_embed_path, tag):
    try:
        # 加载已对齐的文本嵌入向量
        text_vec = np.load(text_embed_path).astype(float).tolist()
        text_uuid = text_embed_path.split("/")[-1].split(".")[0]

        # 加载已对齐的图像嵌入向量
        image_vec = np.load(image_embed_path).astype(float).tolist()
        image_uuid = image_embed_path.split("/")[-1].split(".")[0]

        # 创建或更新节点和关系
        with driver.session() as s:
            # 创建或更新 Patient 节点
            s.run("""
            MERGE (p:Patient {id: $pid})
            SET p.tag = $tag
            """, pid=pid, tag=tag)

            # 创建或更新 VisitText 节点，并建立与 Patient 节点的关系
            s.run("""
            MERGE (vt:VisitText {uuid: $text_uuid})
            SET vt.tag = $tag, vt.embedding = $text_vec
            MERGE (vt)-[:BELONGS_TO]->(p:Patient {id: $pid})
            """, text_uuid=text_uuid, tag=tag, pid=pid, text_vec=text_vec)

            # 创建或更新 VisitImage 节点，并建立与 Patient 节点的关系
            s.run("""
            MERGE (vi:VisitImage {uuid: $image_uuid})
            SET vi.tag = $tag, vi.embedding = $image_vec
            MERGE (vi)-[:BELONGS_TO]->(p:Patient {id: $pid})
            """, image_uuid=image_uuid, tag=tag, pid=pid, image_vec=image_vec)

            # 创建新的关系表示文本和图像的联系
            s.run("""
            MATCH (vt:VisitText {uuid: $text_uuid}), (vi:VisitImage {uuid: $image_uuid})
            MERGE (vt)-[:RELATED_TO_IMAGE]->(vi)
            """, text_uuid=text_uuid, image_uuid=image_uuid)

    except Exception as e:
        print(f"Error importing patient {pid} with files {text_embed_path} and {image_embed_path}: {e}")

def run_import(patient_id, tag):
    # 获取指定 patientId 的所有文本和图像嵌入向量文件路径
    text_embed_paths = glob.glob(f"data/{patient_id}/text/*.npy")
    image_embed_paths = glob.glob(f"data/{patient_id}/image/*.npy")

    # 处理每一对文本和图像文件
    for text_path, image_path in zip(text_embed_paths, image_embed_paths):
        pid = text_path.split("/")[-1].split("_")[0]
        import_patient(pid, text_path, image_path, tag)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python neo4j_import.py <patientId> <tag> <mode>")
        sys.exit(1)

    patient_id = sys.argv[1]
    tag = int(sys.argv[2])

    run_import(patient_id, tag)
