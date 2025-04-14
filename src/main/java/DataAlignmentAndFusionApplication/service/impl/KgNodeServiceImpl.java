package DataAlignmentAndFusionApplication.service.impl;

import DataAlignmentAndFusionApplication.algorithm.data.GraphNode;
import DataAlignmentAndFusionApplication.mapper.KgNodeDetailMapper;
import DataAlignmentAndFusionApplication.model.entity.KgNodeDetail;
import DataAlignmentAndFusionApplication.service.KgNodeService;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class KgNodeServiceImpl implements KgNodeService {
//    private final KgNodeDetailMapper nodeDetailMapper;
    @Override
    public void saveNodes(List<GraphNode> graphNodes, String graphId) {
        List<KgNodeDetail> details = graphNodes.stream()
                .map(node -> convertToDetail(node, graphId))
                .collect(Collectors.toList());

//        nodeDetailMapper.insertBatch(details);
    }

    @Override
    public KgNodeDetail convertToDetail(GraphNode node, String graphId) {
        KgNodeDetail detail = new KgNodeDetail();
        detail.setNodeId(node.getId());
        detail.setGraphId(graphId);
        detail.setNodeType(node.getLabel());
        detail.setProperties(node.getProperties());
        detail.setCreateTime(LocalDateTime.now());
        return detail;
    }

    @Override
    public Map<String, Integer> getNodeTypeDistribution(String graphId) {
//        return nodeDetailMapper.countNodesByType(graphId).stream()
//                .collect(Collectors.toMap(
//                        m -> (String) m.get("node_type"),
//                        m -> ((Long) m.get("count")).intValue()
//                ));
        return Map.of();

    }
}
