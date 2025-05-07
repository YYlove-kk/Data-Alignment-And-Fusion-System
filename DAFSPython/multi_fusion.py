import sys

import numpy as np
from neo4j import GraphDatabase
from pathlib import Path

# 连接到 Neo4j 数据库
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "12345678"))

def import_patient(pid, text_embed_path, image_embed_path, tag, institution):
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

            # 构建版本化 Text 节点
            s.run("""
            MERGE (vt:Text {uuid: $text_uuid, tag: $tag})
            SET vt.embedding = $text_vec
            MERGE (vt)-[:BELONGS_TO]->(p:Patient {id: $pid, tag: $tag})
            """, text_uuid=text_uuid, tag=tag, pid=pid, text_vec=text_vec)

            # 构建版本化 Image 节点
            s.run("""
            MERGE (vi:Image {uuid: $image_uuid, tag: $tag})
            SET vi.embedding = $image_vec
            MERGE (vi)-[:BELONGS_TO]->(p:Patient {id: $pid, tag: $tag})
            """, image_uuid=image_uuid, tag=tag, pid=pid, image_vec=image_vec)

            # 构建版本化 Institution 节点
            s.run("""
            MERGE (ins:Institution {name: $name, tag: $tag})
            SET ins.name = $name
            MERGE (vi)-[:FROM]->(ins:Institution {name: $name, tag: $tag})
            MERGE (vt)-[:FROM]->(ins:Institution {name: $name, tag: $tag})
            """, name=institution, tag=tag)


    except Exception as e:
        print(f"Error importing patient {pid} with files {text_embed_path} and {image_embed_path}: {e}")

def run_import(pid,institution):
    text_path = base_dir / f"{pid}_z_t.npy"
    image_path = base_dir / f"{pid}_z_i.npy"
    if text_path.exists() and image_path.exists():
        import_patient(pid, text_path, image_path, tag, institution)
    else:
        print(f"Missing file(s) for {pid}")

def build_similarity_edges(model_path: str, tag: int, threshold: float = 0.7):
    import torch
    from model.model_han import HAN

    # 加载模型
    model = HAN(in_size=512, hidden_size=128, out_size=1)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # 获取所有 Text 和 Image 节点的嵌入向量
    with driver.session() as session:
        text_nodes = session.run("MATCH (t:Text {tag: $tag}) RETURN t.uuid AS uuid, t.embedding AS embedding", tag=tag)
        image_nodes = session.run("MATCH (i:Image {tag: $tag}) RETURN i.uuid AS uuid, i.embedding AS embedding", tag=tag)

        texts = [(record["uuid"], record["embedding"]) for record in text_nodes]
        images = [(record["uuid"], record["embedding"]) for record in image_nodes]

        for text_uuid, text_vec in texts:
            text_tensor = torch.tensor(text_vec, dtype=torch.float32).unsqueeze(0)
            for image_uuid, image_vec in images:
                image_tensor = torch.tensor(image_vec, dtype=torch.float32).unsqueeze(0)

                with torch.no_grad():
                    logit = model(text_tensor, image_tensor)
                    prob = torch.sigmoid(logit).item()

                if prob >= threshold:
                    session.run("""
                    MATCH (t:Text {uuid: $text_uuid, tag: $tag})
                    MATCH (i:Image {uuid: $image_uuid, tag: $tag})
                    MERGE (t)-[:MULTI_MODAL_SIMILAR {score: $score, tag: $tag}]->(i)
                    """, text_uuid=text_uuid, image_uuid=image_uuid, tag=tag, score=prob)

        print("Similarity edge creation complete.")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python multi_fusion.py <patientIds> <tag> <institution> <model>")
        sys.exit(1)

    patient_ids = sys.argv[1].split(',')
    tag = int(sys.argv[2])
    institution = sys.argv[3]
    model_path = sys.argv[4]
    base_dir = Path("data/align/reduce")

    for pid in patient_ids:
        run_import(pid, institution)

    # 调用构建多模态相似边的方法
    build_similarity_edges(model_path, tag)
