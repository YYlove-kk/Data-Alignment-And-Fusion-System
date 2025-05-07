import sys

from py2neo import Graph
import networkx as nx
import numpy as np

# 连接到Neo4j数据库
def connect_to_neo4j(uri, user, password):
    graph = Graph(uri, auth=(user, password))
    return graph

# 获取指定标签的节点和边
def get_graph_from_tag(graph, tag):
    query = f"MATCH (n:{tag})-[r]->(m) RETURN n, r, m"
    result = graph.run(query)

    G = nx.Graph()  # 创建无向图

    for record in result:
        node1 = record['n']
        node2 = record['m']

        # 添加节点和边到图中
        G.add_node(node1['id'])  # 假设节点有唯一的 'id' 属性
        G.add_node(node2['id'])
        G.add_edge(node1['id'], node2['id'])

    return G

# 计算平均度
def calculate_average_degree(G):
    degrees = [deg for node, deg in G.degree()]
    return np.mean(degrees)

# 计算平均聚类系数
def calculate_average_clustering_coefficient(G):
    return nx.average_clustering(G)

# 计算连通分量数量
def calculate_connected_components(G):
    return nx.number_connected_components(G)

# 主函数，评估图谱结构
def evaluate_graph_structure(graph, tag):
    G = get_graph_from_tag(graph, tag)

    # 计算评估指标
    avg_degree = calculate_average_degree(G)
    avg_clustering = calculate_average_clustering_coefficient(G)
    num_components = calculate_connected_components(G)

    # 输出评估结果
    print(f"评估结果 for tag: {tag}")
    print(f"平均度: {avg_degree:.4f}")
    print(f"平均聚类系数: {avg_clustering:.4f}")
    print(f"连通分量数量: {num_components}")

if __name__ == "__main__":
    # 设置Neo4j数据库的连接参数
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "12345678"
    # 连接到Neo4j数据库
    graph = connect_to_neo4j(uri, user, password)

    if len(sys.argv) != 2:
        print("Usage: python evaluate_graph.py <tag>")
        sys.exit(1)

    tag = int(sys.argv[1])

    # 执行评估
    evaluate_graph_structure(graph, tag)
