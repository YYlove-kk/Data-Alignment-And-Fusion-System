import sys
import subprocess

def run_neo4j_import(patient_id, tag):
    try:
        subprocess.run(["python", "util/neo4j_import.py", patient_id, str(tag)], check=True)
        print("neo4j_import.py 执行成功")
    except subprocess.CalledProcessError as e:
        print(f"neo4j_import.py 执行失败: {e}")

def run_hnsw_builder(patient_id, tag):
    try:
        subprocess.run(["python", "util/hnsw_builder.py", patient_id, str(tag)], check=True)
        print("hnsw_builder.py 执行成功")
    except subprocess.CalledProcessError as e:
        print(f"hnsw_builder.py 执行失败: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python run_fusion.py <patient_id> <tag>")
        sys.exit(1)
    patient_id = sys.argv[1]
    tag = int(sys.argv[2])

    run_neo4j_import(patient_id, tag)
    run_hnsw_builder(patient_id, tag)