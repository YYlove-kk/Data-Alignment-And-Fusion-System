import os
import sys

import hnswlib
import numpy as np
from neo4j import GraphDatabase
from pathlib import Path
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    from model.model_han import AttentionHAN

    # 加载模型
    model = AttentionHAN(in_size=256, hidden_size=128, out_size=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
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

# 基于余弦相似度的对齐函数
def cosine_align(vectors):
    num_vectors = len(vectors)
    similarity_matrix = np.zeros((num_vectors, num_vectors))
    for i in range(num_vectors):
        for j in range(num_vectors):
            similarity_matrix[i, j] = np.dot(vectors[i], vectors[j]) / (np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j]))
    # 选择参考向量
    reference_vector = np.mean(vectors, axis=0)
    aligned_vectors = []
    for i in range(num_vectors):
        aligned_vector = vectors[i] * (np.dot(vectors[i], reference_vector) / np.linalg.norm(reference_vector) ** 2)
        aligned_vectors.append(aligned_vector)
    return aligned_vectors

# 构建单模态相似边
def create_single_modal_edges(p, index_to_uuid, driver, tag, is_text):
    node_label = "Text" if is_text else "Image"
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

def single_edges(text_embed_paths, image_embed_paths, dim, driver, tag):
    print("构建文本单模态边...")
    # 文本单模态对齐
    text_vectors = [np.load(path).astype(float) for path in text_embed_paths]
    aligned_text_vectors = cosine_align(text_vectors)

    p_text = hnswlib.Index(space='cosine', dim=dim)
    p_text.init_index(max_elements=len(text_embed_paths), ef_construction=200, M=16)
    text_index_map = {}
    for i, vector in enumerate(aligned_text_vectors):
        p_text.add_items(vector.reshape(1, -1))
        uuid = text_embed_paths[i].split("/")[-1].split(".")[0]
        text_index_map[i] = uuid
    create_single_modal_edges(p_text, text_index_map, driver, tag, is_text=True)

    print("构建图像单模态边...")
    # 图像单模态对齐
    image_vectors = [np.load(path).astype(float) for path in image_embed_paths]
    aligned_image_vectors = cosine_align(image_vectors)

    p_image = hnswlib.Index(space='cosine', dim=dim)
    p_image.init_index(max_elements=len(image_embed_paths), ef_construction=200, M=16)
    image_index_map = {}
    for i, vector in enumerate(aligned_image_vectors):
        p_image.add_items(vector.reshape(1, -1))
        uuid = image_embed_paths[i].split("/")[-1].split(".")[0]
        image_index_map[i] = uuid
    create_single_modal_edges(p_image, image_index_map, driver, tag, is_text=False)

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python multi_fusion.py <patientIds> <tag> <institution> <model>")
        sys.exit(1)

    patient_ids = sys.argv[1].split(',')
    tag = int(sys.argv[2])
    institution = sys.argv[3]
    model_name = sys.argv[4]
    model_path = "DAFSPython/han" + model_name
    base_dir = Path("data/align/reduce")
    dim = 256

    for pid in patient_ids:
        run_import(pid, institution)

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

        single_edges(text_embed_paths, image_embed_paths, dim, driver, tag)
    except Exception as e:
        print(f"hnsw_builder 运行失败 ❌: {e}")


    # 调用构建多模态相似边的方法
    build_similarity_edges(model_path, tag)
