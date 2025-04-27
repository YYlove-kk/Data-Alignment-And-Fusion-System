package DataAlignmentAndFusionApplication.service;
import DataAlignmentAndFusionApplication.model.dto.ConstructionDTO;
import DataAlignmentAndFusionApplication.model.dto.FusionDTO;
import DataAlignmentAndFusionApplication.model.vo.*;

import java.util.List;
import java.util.Map;

public interface FusionService {
    //图谱构建
    GraphVO buildKnowledgeGraph(ConstructionDTO dto);
    PageVO<ConstructionRecordVO> getConstructionHistory(Long userId, Integer page, Integer size);

    //图谱融合
    FusionVO fuseGraphs(FusionDTO dto);
//    void saveFusionResult(FusionResult result);

    //节点管理
//    void batchSaveNodes(List<GraphNode> nodes, String graphId);
    NodeDetailVO getNodeDetails(String nodeId, Long graphId);
    Map<String, Integer> getNodeTypeDistribution(String graphId);

    //可视化
    GraphVisualizationVO getGraphVisualization(Long graphId);
}
