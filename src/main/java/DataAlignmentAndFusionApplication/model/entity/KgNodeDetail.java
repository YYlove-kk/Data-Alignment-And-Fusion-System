package DataAlignmentAndFusionApplication.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import jakarta.validation.constraints.NotBlank;
import lombok.Data;

import java.time.LocalDateTime;
import java.util.Map;

@Data
@TableName("kg_node_detail")
public class KgNodeDetail {
    @TableId(type = IdType.ASSIGN_ID) // 分布式ID生成
    private Long id;

    @NotBlank
    @TableField("node_id")
    private String nodeId;          // 节点全局唯一ID

    @NotBlank
    @TableField("graph_id")
    private String graphId;         // 所属图谱ID

    @NotBlank
    @TableField("node_type")
    private String nodeType;        // 节点类型（如Person/Company）

    private Map<String, Object> properties; // 动态属性

    @TableField("create_time")
    private LocalDateTime createTime;

    @TableField("update_time")
    private LocalDateTime updateTime;
}
