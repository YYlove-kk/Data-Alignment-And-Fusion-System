package DataAlignmentAndFusionApplication.model.vo;

import lombok.Data;

import java.util.List;
import java.util.Map;
// 图谱节点详情VO
@Data
public class NodeDetailVO {
    private String nodeId;
    private String label;
    private String dataType;        // 文本/图像等
    private Map<String, Object> properties;
    private List<RelationVO> relations; // 关联关系
}