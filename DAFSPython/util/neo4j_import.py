import sys

import hnswlib
import numpy as np
from neo4jtest import GraphDatabase
from pathlib import Path

# 连接到 Neo4j 数据库
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "12345678"))

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

def build_cross_modal_index(text_map, image_map):
    combined_map = {}
    # 这里name加后缀标记类型，保证唯一
    for name, vec in text_map.items():
        combined_map[f"{name}_Text"] = vec
    for name, vec in image_map.items():
        combined_map[f"{name}_Image"] = vec
    dim = len(next(iter(combined_map.values())))
    index, index_to_key = build_hnsw_index(combined_map, dim)
    return index, index_to_key, combined_map

def create_cross_modal_edges(index, index_to_key, combined_map, driver, tag, rel_type):
    n = len(index_to_key)
    k = min(n - 1, 20)
    vecs = list(combined_map.values())
    count = 0
    with driver.session() as session:
        for i in range(n):
            query_vec = vecs[i].reshape(1, -1)
            labels, distances = index.knn_query(query_vec, k=k)
            src_key = index_to_key[i]
            src_name, src_label = src_key.rsplit("_", 1)
            for j, neighbor_idx in enumerate(labels[0]):
                if i == neighbor_idx:
                    continue
                dst_key = index_to_key[neighbor_idx]
                dst_name, dst_label = dst_key.rsplit("_", 1)
                # 只建立跨模态边，排除单模态边
                if src_label == dst_label:
                    continue
                sim = 1 - distances[0][j]
                print(f"[{rel_type}] {src_key} -> {dst_key}, similarity = {sim:.4f}")
                session.run(f"""
                    MATCH (a:{src_label} {{name: $src_name, tag: $tag}}), (b:{dst_label} {{name: $dst_name, tag: $tag}})
                    MERGE (a)-[:{rel_type} {{weight: $weight, tag: $tag}}]->(b)
                """, src_name=src_name, dst_name=dst_name, weight=sim, tag=tag)
                count += 1
    print(f"Total cross-modal edges created: {count}")

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



def run_hnsw_builder(tag, mode):
    dim = 256
    try:
        if mode in (0, 1):  # 包含单模态构建
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

        if mode == 0:  # 跨模态构建
            print("构建跨模态边（Text-Image）...")
            text_map = fetch_embeddings(driver, tag, "Text")
            image_map = fetch_embeddings(driver, tag, "Image")
            combined_index, index_to_key, combined_map = build_cross_modal_index(text_map, image_map)
            create_cross_modal_edges(combined_index, index_to_key, combined_map, driver, tag, "MULTI_MODAL_SIMILAR")

    except Exception as e:
        print(f"hnsw_builder 运行失败 ❌: {e}")

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
    tag = 2
    mode = 0
    institution = "影像"
    type = "image"
    base_dir = Path("../../data/align/match")


    for pid in patient_ids:
        run_import(pid,institution,type)
        check_patient_exists(pid, tag)

    run_hnsw_builder(tag, mode)
