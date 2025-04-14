package DataAlignmentAndFusionApplication.model.vo;

import lombok.Data;

import java.time.LocalDateTime;

//融合结果视图对象

@Data
public class FusionVO {
    private String fusionId;
    private Long mainGraphId;
    private Long subGraphId;
    private Integer addedNodes;     // 新增节点数
    private Integer mergedRelations; // 合并关系数
    private String visualizationUrl; // 融合差异可视化URL
    private LocalDateTime createTime;
}