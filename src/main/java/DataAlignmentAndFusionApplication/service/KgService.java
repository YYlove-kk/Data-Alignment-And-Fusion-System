package DataAlignmentAndFusionApplication.service;

import DataAlignmentAndFusionApplication.algorithm.data.FusionResult;
import DataAlignmentAndFusionApplication.model.DTO.ConstructionDTO;
import DataAlignmentAndFusionApplication.model.DTO.FusionDTO;
import DataAlignmentAndFusionApplication.model.vo.*;
import org.springframework.transaction.annotation.Transactional;

public interface KgService {
    /**
     * 构建知识图谱
     */
    GraphVO buildKnowledgeGraph(ConstructionDTO dto);

    /**
     * 融合图谱
     */
    FusionVO fuseGraphs(FusionDTO dto);

    /**
     * 获取图谱可视化数据
     */
    GraphVisualizationVO getGraphVisualization(Long graphId);

    /**
     * 查询节点详情
     */
    NodeDetailVO getNodeDetails(String nodeId, Long graphId);

    /**
     * 历史记录
     * @param userId
     * @param page
     * @param size
     * @return
     */
    PageVO<ConstructionRecordVO> getConstructionHistory(Long userId, Integer page, Integer size);

    /**
     * 保存图谱融合结果
     * @param result 融合结果数据
     */
    @Transactional
    void saveFusionResult(FusionResult result);
}
