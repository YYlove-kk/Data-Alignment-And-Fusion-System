package DataAlignmentAndFusionApplication.service.impl;

import DataAlignmentAndFusionApplication.algorithm.data.FusionResult;
import DataAlignmentAndFusionApplication.algorithm.data.GraphNode;
import DataAlignmentAndFusionApplication.model.dto.ConstructionDTO;
import DataAlignmentAndFusionApplication.model.dto.FusionDTO;
import DataAlignmentAndFusionApplication.model.vo.*;
import DataAlignmentAndFusionApplication.service.FusionService;

import java.util.List;
import java.util.Map;

public class FusionServiceImpl implements FusionService {


    @Override
    public GraphVO buildKnowledgeGraph(ConstructionDTO dto) {
        return null;
    }

    @Override
    public PageVO<ConstructionRecordVO> getConstructionHistory(Long userId, Integer page, Integer size) {
        return null;
    }

    @Override
    public FusionVO fuseGraphs(FusionDTO dto) {
        return null;
    }

//    @Override
//    public void saveFusionResult(FusionResult result) {
//
//    }
//
//    @Override
//    public void batchSaveNodes(List<GraphNode> nodes, String graphId) {
//
//    }

    @Override
    public NodeDetailVO getNodeDetails(String nodeId, Long graphId) {
        return null;
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

    @Override
    public GraphVisualizationVO getGraphVisualization(Long graphId) {
        return null;
    }
}
