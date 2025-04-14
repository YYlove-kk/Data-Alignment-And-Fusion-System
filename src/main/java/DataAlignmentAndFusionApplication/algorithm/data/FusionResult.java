package DataAlignmentAndFusionApplication.algorithm.data;

import DataAlignmentAndFusionApplication.model.entity.KgFusionRecord;
import lombok.Data;

import java.time.LocalDateTime;
import java.util.List;

@Data
public class FusionResult {
    // 运行时数据
    private transient GraphData mergedGraph;
    private List<ConflictDetail> conflicts;

    // 融合元数据（需补充字段）
    private Long mainGraphId;
    private Long subGraphId;
    private String strategyType;

    // 转换方法
    public KgFusionRecord toRecord() {
        KgFusionRecord record = new KgFusionRecord();
        record.setMainGraphId(this.mainGraphId);
        record.setSubGraphId(this.subGraphId);
        record.setStrategyType(this.strategyType);
        record.setConflictCount(this.conflicts != null ? this.conflicts.size() : 0);
        record.setNewNodeCount(calculateNewNodes());
//        record.setSnapshot(buildSnapshot());
        record.setCreateTime(LocalDateTime.now());
        return record;
    }

    private Integer calculateNewNodes() {
        // 实际实现：对比新旧图谱的节点差异
        return this.mergedGraph != null ?
                this.mergedGraph.getNodes().size() : 0;
    }

    private String buildSnapshot() {
        if (this.mergedGraph == null) return "{}";

        return null;
    }
}