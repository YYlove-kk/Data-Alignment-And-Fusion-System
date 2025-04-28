from neo4j import GraphDatabase
import numpy as np
import glob


driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "12345678"))

def import_patient(pid, embed_path):
    try:
        # 加载嵌入向量
        vec = np.load(embed_path).astype(float).tolist()
        uuid = embed_path.split("/")[-1].split(".")[0]  # 从路径中提取UUID
        # 创建或更新节点和关系
        with driver.session() as s:
            s.run("""
            MERGE (p:Patient {id:$pid})
            MERGE (v:Visit {uuid:$uuid})-[:BELONGS_TO]->(p)
            SET v.embedding = $vec,
                v.id = $uuid
            """, pid=pid, uuid=uuid, vec=vec)
    except Exception as e:
        # 错误处理
        print(f"Error importing patient {pid} with file {embed_path}: {e}")

def main():
    # 查找所有嵌入向量文件并导入数据
    embed_paths = glob.glob("fusion_embed/*.npy")
    for path in embed_paths:
        pid = path.split("/")[-1].split("_")[0]
        import_patient(pid, path)

if __name__ == "__main__":
    main()