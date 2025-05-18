import sys

import hnswlib
import numpy as np
import torch

from model.simple_gnn import SimpleGNN
from neo4jtest import GraphDatabase
from pathlib import Path

# 连接到 Neo4j 数据库
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "12345678"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def check_patient_exists(pid, tag):
    with driver.session() as session:
        result = session.run("""
            MATCH (p:Patient {id: $pid, tag: $tag}) RETURN p
        """, pid=pid, tag=tag)
        record = result.single()
        if record:
            print(f"Patient {pid} exists in Neo4j.")
        else:
            print(f"Patient {pid} NOT found in Neo4j.")


def import_patient(pid, text_embed_path, image_embed_path, tag, institution, type):
    try:
        # 加载已对齐的文本嵌入向量
        text_vec = np.load(text_embed_path).astype(float).tolist()
        text_name = text_embed_path.stem

        # 加载已对齐的图像嵌入向量
        image_vec = np.load(image_embed_path).astype(float).tolist()
        image_name = image_embed_path.stem
        # 创建或更新节点和关系
        with driver.session() as s:
            # 构建版本化 Patient 节点
            s.run("""
            MERGE (p:Patient {name: $pid, tag: $tag})
            """, pid=pid, tag=tag)

            # 构建版本化 Text 节点
            s.run("""
            MERGE (vt:Text {name: $text_name, tag: $tag})
            SET vt.embedding = $text_vec
            WITH vt
            MATCH (p:Patient {name: $pid, tag: $tag})
            MERGE (vt)-[:BELONGS_TO {tag: $tag}]->(p)
            """, tag=tag, pid=pid, text_vec=text_vec,text_name = text_name)

            # 构建版本化 Image 节点
            s.run("""
            MERGE (vi:Image {name: $image_name, tag: $tag})
            SET vi.embedding = $image_vec
            WITH vi
            MATCH (p:Patient {name: $pid, tag: $tag})
            MERGE (vi)-[:BELONGS_TO {tag: $tag}]->(p)
            """, tag=tag, pid=pid, image_vec=image_vec,image_name = image_name)

            if type == "text":
                # 构建版本化 Institution 节点（文本）
                s.run("""
                MERGE (ins:Institution {name: $name, tag: $tag})
                SET ins.name = $name
                WITH ins
                MATCH (vt:Text {name: $text_name, tag: $tag})
                MERGE (vt)-[:FROM {tag: $tag}]->(ins)
                """, name=institution, tag=tag, text_name=text_name)
            else:
                # 构建版本化 Institution 节点（图像）
                s.run("""
                MERGE (ins:Institution {name: $name, tag: $tag})
                SET ins.name = $name
                WITH ins
                MATCH (vi:Image {name: $image_name, tag: $tag})
                MERGE (vi)-[:FROM {tag: $tag}]->(ins)
                """, name=institution, tag=tag, image_name=image_name)
    except Exception as e:
        print(f"Error importing patient {pid} with files {text_embed_path} and {image_embed_path}: {e}")

def run_import(pid,institution,type):
    text_path = base_dir / "txt" /f"{pid}_z_t.npy"
    image_path = base_dir / "img" /f"{pid}_z_i.npy"
    if text_path.exists() and image_path.exists():
        import_patient(pid, text_path, image_path, tag, institution, type)
    else:
        print(f"Missing file(s) for {pid}")

def fetch_embeddings(driver, tag, label):

    query = f"""
    MATCH (n:{label} {{tag: $tag}})
    RETURN n.name AS name, n.embedding AS embedding
    """
    name_vec_map = {}
    with driver.session() as session:
        results = session.run(query, tag=tag)
        for record in results:
            name = record["name"]
            embedding = record["embedding"]
            name_vec_map[name] = np.array(embedding, dtype=float)
    return name_vec_map

def build_hnsw_index(name_vec_map, dim):
    index = hnswlib.Index(space='cosine', dim=dim)
    max_elements = len(name_vec_map)
    index.init_index(max_elements=max_elements, ef_construction=200, M=16)
    index_to_name = {}
    for i, (name, vec) in enumerate(name_vec_map.items()):
        vec = vec.reshape(-1)  # 保证一维向量
        index.add_items(vec, i)  # 指定id为 i
        index_to_name[i] = name
    return index, index_to_name

def create_similar_edges(index, index_to_name, name_vec_map, driver, tag, label, rel_type):
    n = len(index_to_name)
    k = min(n - 1, 20)
    vecs = list(name_vec_map.values())
    count = 0
    with driver.session() as session:
        for i in range(n):
            query_vec = vecs[i].reshape(1, -1)
            labels, distances = index.knn_query(query_vec, k=k)
            for j, neighbor_idx in enumerate(labels[0]):
                if i == neighbor_idx:
                    continue
                name1 = index_to_name[i]
                name2 = index_to_name[neighbor_idx]
                sim = 1 - distances[0][j]
                print(f"[{rel_type}] {name1} -> {name2}, similarity = {sim:.4f}")
                if sim >= 0:
                    session.run(f"""
                    MATCH (a:{label} {{name: $name1, tag: $tag}}), (b:{label} {{name: $name2, tag: $tag}})
                    MERGE (a)-[:{rel_type} {{weight: $weight, tag: $tag}}]->(b)
                    """, name1=name1, name2=name2, weight=sim, tag=tag)
                    count += 1
    print(f"Total edges created: {count}")

def cosine_align(vectors):
    num_vectors = len(vectors)
    similarity_matrix = np.zeros((num_vectors, num_vectors))
    for i in range(num_vectors):
        for j in range(num_vectors):
            similarity_matrix[i, j] = np.dot(vectors[i], vectors[j]) / (np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j]))
    reference_vector = np.mean(vectors, axis=0)
    aligned_vectors = []
    for i in range(num_vectors):
        aligned_vector = vectors[i] * (np.dot(vectors[i], reference_vector) / np.linalg.norm(reference_vector) ** 2)
        aligned_vectors.append(aligned_vector)
    return aligned_vectors


def run_hnsw_builder(tag, ):
    dim = 256
    try:

            print("读取 Text 向量...")
            text_map = fetch_embeddings(driver, tag, "Text")  # {name: vector}
            print("对齐 Text 向量...")
            aligned_text_map = {}
            text_names = list(text_map.keys())
            text_vectors = [text_map[name] for name in text_names]
            aligned_text_vectors = cosine_align(text_vectors)
            for name, vec in zip(text_names, aligned_text_vectors):
                aligned_text_map[name] = vec
            print("构建 Text HNSW 索引...")
            text_index, text_index_to_name = build_hnsw_index(aligned_text_map, dim)
            print("构建 Text 单模态边...")
            create_similar_edges(text_index, text_index_to_name, aligned_text_map, driver, tag, "Text", "SINGLE_MODAL_SIMILAR")

            print("读取 Image 向量...")
            image_map = fetch_embeddings(driver, tag, "Image")  # {name: vector}
            print("对齐 Image 向量...")
            aligned_image_map = {}
            image_names = list(image_map.keys())
            image_vectors = [image_map[name] for name in image_names]
            aligned_image_vectors = cosine_align(image_vectors)
            for name, vec in zip(image_names, aligned_image_vectors):
                aligned_image_map[name] = vec
            print("构建 Image HNSW 索引...")
            image_index, image_index_to_name = build_hnsw_index(aligned_image_map, dim)
            print("构建 Image 单模态边...")
            create_similar_edges(image_index, image_index_to_name, aligned_image_map, driver, tag, "Image", "SINGLE_MODAL_SIMILAR")

    except Exception as e:
        print(f"hnsw_builder 运行失败 ❌: {e}")

def build_similarity_edges(model_path: str, tag: int, threshold: float = 0.7):
    from model.model_han import AttentionHAN

    # 加载模型
    # model = AttentionHAN(in_size=256, hidden_size=128, out_size=1, num_heads=4, threshold=0.6)


    model = SimpleGNN(in_size=256, hidden_size=128, out_size=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # 设置为评估模式

    # 获取所有 Text 和 Image 节点的嵌入向量
    with driver.session() as session:
        text_nodes = session.run(
            "MATCH (t:Text {tag: $tag}) RETURN t.name AS name, t.embedding AS embedding", tag=tag)
        image_nodes = session.run(
            "MATCH (i:Image {tag: $tag}) RETURN i.name AS name, i.embedding AS embedding", tag=tag)

        texts = [(record["name"], record["embedding"]) for record in text_nodes]
        images = [(record["name"], record["embedding"]) for record in image_nodes]

        for text_name, text_vec in texts:
            text_tensor = torch.tensor(text_vec, dtype=torch.float32).unsqueeze(0).to(device)
            for image_name, image_vec in images:
                image_tensor = torch.tensor(image_vec, dtype=torch.float32).unsqueeze(0).to(device)

                with torch.no_grad():
                    logit = model(text_tensor, image_tensor)
                    prob = torch.sigmoid(logit).item()
                    print(prob)

                if prob >= threshold:
                    session.run("""
                        MATCH (t:Text {name: $text_name, tag: $tag})
                        MATCH (i:Image {name: $image_name, tag: $tag})
                        MERGE (t)-[:MULTI_MODAL_SIMILAR {score: $score, tag: $tag}]->(i)
                    """, text_name=text_name, image_name=image_name, tag=tag, score=prob)

        print("Similarity edge creation complete.")


if __name__ == "__main__":
    # if len(sys.argv) != 5:
    #     print("Usage: python neo4j_import.py <patientIdsStr> <tag> <mode> <institution> <type>")
    #     sys.exit(1)

    ids = "pacs01CT201702170412,pacs01MR201210150661,pacs01MR201407260162,pacs01MR201703060046"
    # patient_ids = sys.argv[1].split(',')
    # tag = int(sys.argv[2])
    # institution = sys.argv[4]
    # type = sys.argv[5]

    patient_ids = ids.split(',')
    tag = 6
    institution = "文本"
    type = "text"
    base_dir = Path("../data/align/match")


    for pid in patient_ids:
        run_import(pid,institution,type)
        check_patient_exists(pid, tag)

    run_hnsw_builder(tag)
    build_similarity_edges('gnn/han_epoch100.pt', tag, threshold=0.7)

