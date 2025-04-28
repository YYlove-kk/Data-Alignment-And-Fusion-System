import hnswlib
import numpy as np

def build_hnsw_index(embed_paths, dim=512):
    p = hnswlib.Index(space='cosine', dim=dim)
    p.init_index(max_elements=len(embed_paths), ef_construction=200, M=16)
    index_to_uuid = {}
    for i, path in enumerate(embed_paths):
        vec = np.load(path).astype(float)
        p.add_items(vec.reshape(1, -1))
        uuid = path.split("/")[-1].split(".")[0]
        index_to_uuid[i] = uuid
    return p, index_to_uuid

def create_similar_edges(p, index_to_uuid, driver):
    with driver.session() as s:
        for i in range(len(index_to_uuid)):
            labels, distances = p.knn_query(p.get_items([i])[0], k=20)
            for j, label in enumerate(labels[0]):
                if i != label:
                    sim = 1 - distances[0][j]
                    if sim >= 0.7:
                        uuid1, uuid2 = index_to_uuid[i], index_to_uuid[label]
                        s.run("""
                        MATCH (v1:Visit {uuid:$uuid1}), (v2:Visit {uuid:$uuid2})
                        MERGE (v1)-[r:SIMILAR {weight:$weight}]->(v2)
                        """, uuid1=uuid1, uuid2=uuid2, weight=sim)
