import sys

import numpy as np
from neo4j import GraphDatabase
from pathlib import Path

# 连接到 Neo4j 数据库
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "12345678"))

def import_patient(pid, text_embed_path, image_embed_path, tag):
    try:
        # 加载已对齐的文本嵌入向量
        text_vec = np.load(text_embed_path).astype(float).tolist()
        text_uuid = text_embed_path.stem

        # 加载已对齐的图像嵌入向量
        image_vec = np.load(image_embed_path).astype(float).tolist()
        image_uuid = image_embed_path.stem

        # 创建或更新节点和关系
        with driver.session() as s:
            # 构建版本化 Patient 节点
            s.run("""
            MERGE (p:Patient {id: $pid, tag: $tag})
            """, pid=pid, tag=tag)

            # 构建版本化 VisitText 节点
            s.run("""
            MERGE (vt:VisitText {uuid: $text_uuid, tag: $tag})
            SET vt.embedding = $text_vec
            MERGE (vt)-[:BELONGS_TO]->(p:Patient {id: $pid, tag: $tag})
            """, text_uuid=text_uuid, tag=tag, pid=pid, text_vec=text_vec)

            # 构建版本化 VisitImage 节点
            s.run("""
            MERGE (vi:VisitImage {uuid: $image_uuid, tag: $tag})
            SET vi.embedding = $image_vec
            MERGE (vi)-[:BELONGS_TO]->(p:Patient {id: $pid, tag: $tag})
            """, image_uuid=image_uuid, tag=tag, pid=pid, image_vec=image_vec)

            # 建立文本和图像的联系
            s.run("""
            MATCH (vt:VisitText {uuid: $text_uuid, tag: $tag}), (vi:VisitImage {uuid: $image_uuid, tag: $tag})
            MERGE (vt)-[:RELATED_TO_IMAGE]->(vi)
            """, text_uuid=text_uuid, image_uuid=image_uuid, tag=tag)

    except Exception as e:
        print(f"Error importing patient {pid} with files {text_embed_path} and {image_embed_path}: {e}")

def run_import(pid):
    text_path = base_dir / pid / f"{pid}_z_t.npy"
    image_path = base_dir / pid / f"{pid}_z_i.npy"
    if text_path.exists() and image_path.exists():
        import_patient(pid, text_path, image_path, tag)
    else:
        print(f"Missing file(s) for {pid}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python neo4j_import.py <patientId> <tag> <mode>")
        sys.exit(1)

    pid = sys.argv[1]
    tag = int(sys.argv[2])
    base_dir = Path("data/align/reduce")

    run_import(pid)


