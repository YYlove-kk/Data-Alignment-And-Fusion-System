package DataAlignmentAndFusionApplication.algorithm.data;

import lombok.Data;

@Data
public class ConflictDetail {
    private String nodeId1;      // 冲突节点1
    private String nodeId2;      // 冲突节点2
    private String conflictType; // 属性冲突/关系冲突等
    private String resolution;  // 解决方式（自动/手动）
    private String description; // 冲突描述
}
