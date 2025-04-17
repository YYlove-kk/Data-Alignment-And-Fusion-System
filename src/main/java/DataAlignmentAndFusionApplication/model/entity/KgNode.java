package DataAlignmentAndFusionApplication.model.entity;

import com.baomidou.mybatisplus.annotation.*;
import jakarta.validation.constraints.NotBlank;
import lombok.Data;

import java.time.LocalDateTime;
import java.util.Map;

@Data
@TableName("kg_node")
public class KgNode {
    @TableId(type = IdType.ASSIGN_ID)
    private String nodeId;          // 直接使用业务ID，避免多级映射

    private String graphId;
    private String nodeType;

    // 将常用查询属性提取为独立字段
    private String name;            // 从properties中提取
    private String dataSource;      // 来源系统
    private Integer version = 1;    // 版本控制

//    @Column(typeHandler = JsonTypeHandler.class)
    private Map<String, Object> extendedProps; // 非核心属性

    // 自动维护时间
    @TableField(fill = FieldFill.INSERT)
    private LocalDateTime createTime;
    @TableField(fill = FieldFill.INSERT_UPDATE)
    private LocalDateTime updateTime;
}
