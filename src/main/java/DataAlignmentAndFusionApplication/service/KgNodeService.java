package DataAlignmentAndFusionApplication.service;

import DataAlignmentAndFusionApplication.algorithm.data.GraphNode;
import DataAlignmentAndFusionApplication.model.entity.KgNodeDetail;

import java.util.List;
import java.util.Map;

public interface KgNodeService {

    // 批量保存节点
    void saveNodes(List<GraphNode> graphNodes, String graphId);

    KgNodeDetail convertToDetail(GraphNode node, String graphId);

    // 查询图谱节点分布
    Map<String, Integer> getNodeTypeDistribution(String graphId);
}
