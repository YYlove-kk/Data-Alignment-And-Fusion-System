import networkx as nx

# 定义节点和边的数据
nodes = [
    { "id": "pacs01CT201702170412", "type": "Patient", "nodeDetail": {} },
    { "id": "pacs01CT201702170416", "type": "Patient", "nodeDetail": {} },
    { "id": "pacs01CT201702170413", "type": "Patient", "nodeDetail": {} },
    { "id": "pacs01CT201702170415", "type": "Patient", "nodeDetail": {} },
    { "id": "诊断记录表", "type": "Institution", "nodeDetail": {} },
    { "id": "dcm影像记录", "type": "Institution", "nodeDetail": {} },
    { "id": "pacs01CT201702170412_z_t", "type": "Text", "nodeDetail": { "textFile": "pacs01CT201702170412_z_t.npy", "imageFile": None } },
    { "id": "pacs01CT201702170413_z_t", "type": "Text", "nodeDetail": { "textFile": "pacs01CT201702170413_z_t.npy", "imageFile": None } },
    { "id": "pacs01CT201702170415_z_t", "type": "Text", "nodeDetail": { "textFile": "pacs01CT201702170415_z_t.npy", "imageFile": None } },
    { "id": "pacs01CT201702170416_z_t", "type": "Text", "nodeDetail": { "textFile": "pacs01CT201702170416_z_t.npy", "imageFile": None } },
    { "id": "pacs01CT201702170412_z_i", "type": "Image", "nodeDetail": { "textFile": None, "imageFile": "pacs01CT201702170412_z_i.npy" } },
    { "id": "pacs01CT201702170413_z_i", "type": "Image", "nodeDetail": { "textFile": None, "imageFile": "pacs01CT201702170413_z_i.npy" } },
    { "id": "pacs01CT201702170415_z_i", "type": "Image", "nodeDetail": { "textFile": None, "imageFile": "pacs01CT201702170415_z_i.npy" } },
    { "id": "pacs01CT201702170416_z_i", "type": "Image", "nodeDetail": { "textFile": None, "imageFile": "pacs01CT201702170416_z_i.npy" } }
]

edges = [
    { "source": "pacs01CT201702170412_z_t", "target": "pacs01CT201702170413_z_t", "relations": [{"relation": "TEXT_SIMILAR", "weight": 0.82}] },
    { "source": "pacs01CT201702170415_z_t", "target": "pacs01CT201702170416_z_t", "relations": [{"relation": "TEXT_SIMILAR", "weight": 0.76}] },
    { "source": "pacs01CT201702170412_z_i", "target": "pacs01CT201702170413_z_i", "relations": [{"relation": "IMAGE_SIMILAR", "weight": 0.79}] },
    { "source": "pacs01CT201702170415_z_i", "target": "pacs01CT201702170416_z_i", "relations": [{"relation": "IMAGE_SIMILAR", "weight": 0.73}] },
    { "source": "pacs01CT201702170412_z_t", "target": "pacs01CT201702170412", "relations": [{"relation": "BELONGS_TO", "weight": None}] },
    { "source": "pacs01CT201702170412_z_i", "target": "pacs01CT201702170412", "relations": [{"relation": "BELONGS_TO", "weight": None}] },
    { "source": "pacs01CT201702170413_z_t", "target": "pacs01CT201702170413", "relations": [{"relation": "BELONGS_TO", "weight": None}] },
    { "source": "pacs01CT201702170413_z_i", "target": "pacs01CT201702170413", "relations": [{"relation": "BELONGS_TO", "weight": None}] },
    { "source": "pacs01CT201702170415_z_t", "target": "pacs01CT201702170415", "relations": [{"relation": "BELONGS_TO", "weight": None}] },
    { "source": "pacs01CT201702170415_z_i", "target": "pacs01CT201702170415", "relations": [{"relation": "BELONGS_TO", "weight": None}] },
    { "source": "pacs01CT201702170416_z_t", "target": "pacs01CT201702170416", "relations": [{"relation": "BELONGS_TO", "weight": None}] },
    { "source": "pacs01CT201702170416_z_i", "target": "pacs01CT201702170416", "relations": [{"relation": "BELONGS_TO", "weight": None}] },
    { "source": "pacs01CT201702170412_z_t", "target": "诊断记录表", "relations": [{"relation": "FROM", "weight": None}] },
    { "source": "pacs01CT201702170413_z_t", "target": "诊断记录表", "relations": [{"relation": "FROM", "weight": None}] },
    { "source": "pacs01CT201702170415_z_t", "target": "诊断记录表", "relations": [{"relation": "FROM", "weight": None}] },
    { "source": "pacs01CT201702170416_z_t", "target": "诊断记录表", "relations": [{"relation": "FROM", "weight": None}] },
    { "source": "pacs01CT201702170412_z_i", "target": "dcm影像记录", "relations": [{"relation": "FROM", "weight": None}] },
    { "source": "pacs01CT201702170413_z_i", "target": "dcm影像记录", "relations": [{"relation": "FROM", "weight": None}] },
    { "source": "pacs01CT201702170415_z_i", "target": "dcm影像记录", "relations": [{"relation": "FROM", "weight": None}] },
    { "source": "pacs01CT201702170416_z_i", "target": "dcm影像记录", "relations": [{"relation": "FROM", "weight": None}] },
    { "source": "pacs01CT201702170412_z_t", "target": "pacs01CT201702170415_z_i", "relations": [{ "relation": "CROSS_MODAL_SIMILAR", "weight": 0.73 }] },
    { "source": "pacs01CT201702170412_z_t", "target": "pacs01CT201702170416_z_i", "relations": [{ "relation": "CROSS_MODAL_SIMILAR", "weight": 0.75 }] },
    { "source": "pacs01CT201702170413_z_t", "target": "pacs01CT201702170415_z_i", "relations": [{ "relation": "CROSS_MODAL_SIMILAR", "weight": 0.78 }] },
    { "source": "pacs01CT201702170413_z_t", "target": "pacs01CT201702170416_z_i", "relations": [{ "relation": "CROSS_MODAL_SIMILAR", "weight": 0.76 }] },
    { "source": "pacs01CT201702170415_z_t", "target": "pacs01CT201702170412_z_i", "relations": [{ "relation": "CROSS_MODAL_SIMILAR", "weight": 0.72 }] },
    { "source": "pacs01CT201702170415_z_t", "target": "pacs01CT201702170413_z_i", "relations": [{ "relation": "CROSS_MODAL_SIMILAR", "weight": 0.74 }] },
    { "source": "pacs01CT201702170416_z_t", "target": "pacs01CT201702170412_z_i", "relations": [{ "relation": "CROSS_MODAL_SIMILAR", "weight": 0.79 }] },
    { "source": "pacs01CT201702170416_z_t", "target": "pacs01CT201702170413_z_i", "relations": [{ "relation": "CROSS_MODAL_SIMILAR", "weight": 0.77 }] },
    { "source": "pacs01CT201702170416_z_t", "target": "pacs01CT201702170416_z_i", "relations": [{ "relation": "CROSS_MODAL_SIMILAR", "weight": 0.81 }] }
]

# 创建图对象
G = nx.Graph()

# 添加节点
for node in nodes:
    G.add_node(node["id"])

# 添加边
for edge in edges:
    G.add_edge(edge["source"], edge["target"])

# 计算平均度
average_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()

# 计算平均聚类系数
average_clustering = nx.average_clustering(G)

# 计算连通分量数量
connected_components = nx.number_connected_components(G)

print(f"平均度: {average_degree}")
print(f"平均聚类系数: {average_clustering}")
print(f"连通分量数量: {connected_components}")