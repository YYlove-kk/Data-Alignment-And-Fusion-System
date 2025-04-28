# util/dgl_loader.py
import dgl
import torch
from neo4j import GraphDatabase
from collections import defaultdict

def load_neo4j(uri, user="neo4j", password="pwd"):
    driver = GraphDatabase.driver(uri, auth=(user, password))
    node_info = []  # (node_id, node_type, props)
    edge_info = []  # (source_id, target_id, rel_type)

    # Step 1: 读取所有节点
    with driver.session() as session:
        result = session.run("MATCH (n) RETURN id(n) as node_id, labels(n) as labels, properties(n) as props")
        for record in result:
            node_info.append((record["node_id"], record["labels"][0], record["props"]))

    # Step 2: 读取所有边
    with driver.session() as session:
        result = session.run("MATCH (u)-[r]->(v) RETURN id(u) as source, id(v) as target, type(r) as rel_type")
        for record in result:
            edge_info.append((record["source"], record["target"], record["rel_type"]))

    driver.close()

    # Step 3: 组织节点，按类型分组
    node_id_maps = defaultdict(dict)  # {nodetype: {node_id: local_index}}
    features = defaultdict(list)
    labels = defaultdict(list)
    type_counts = defaultdict(int)

    for node_id, node_type, props in node_info:
        idx = type_counts[node_type]
        node_id_maps[node_type][node_id] = idx
        type_counts[node_type] += 1

        if 'embedding' in props:
            features[node_type].append(torch.tensor(props['embedding'], dtype=torch.float32))
        else:
            features[node_type].append(torch.zeros(512))  # 缺省向量

        if 'label' in props:
            labels[node_type].append(props['label'])

    # Step 4: 组织边
    graph_data = defaultdict(lambda: ([], []))
    for src_id, dst_id, rel_type in edge_info:
        src_type = None
        dst_type = None
        for nt in node_id_maps:
            if src_id in node_id_maps[nt]:
                src_type = nt
            if dst_id in node_id_maps[nt]:
                dst_type = nt
        if src_type and dst_type:
            src_idx = node_id_maps[src_type][src_id]
            dst_idx = node_id_maps[dst_type][dst_id]
            graph_data[(src_type, rel_type, dst_type)][0].append(src_idx)
            graph_data[(src_type, rel_type, dst_type)][1].append(dst_idx)

    # Step 5: 构建异质图
    g = dgl.heterograph({
        (srctype, rel, dsttype): (torch.tensor(srcs), torch.tensor(dsts))
        for (srctype, rel, dsttype), (srcs, dsts) in graph_data.items()
    })

    # Step 6: 加载特征和标签
    for ntype in g.ntypes:
        g.nodes[ntype].data['feat'] = torch.stack(features[ntype])
        if ntype in labels:
            g.nodes[ntype].data['label'] = torch.tensor(labels[ntype])

    return g
