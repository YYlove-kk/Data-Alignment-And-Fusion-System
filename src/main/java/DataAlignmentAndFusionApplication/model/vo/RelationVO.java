package DataAlignmentAndFusionApplication.model.vo;

import lombok.Data;

import java.util.Map;

//关系视图对象
@Data
public class RelationVO {
    private String relationId;
    private String sourceNodeId;
    private String targetNodeId;
    private String type; // 关系类型
    private Map<String, Object> properties; // 关系属性
    private String dataSource; // 来源数据集
}