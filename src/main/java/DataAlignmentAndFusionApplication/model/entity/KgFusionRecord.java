package DataAlignmentAndFusionApplication.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

import java.time.LocalDateTime;

@Data
@TableName("kg_fusion_record")
public class KgFusionRecord {
    @TableId(type = IdType.AUTO)
    private Long id;

    @NotNull
    private Long mainGraphId;       // 主图谱ID

    @NotNull
    private Long subGraphId;       // 被融合的子图谱ID

    @NotNull
    private String strategyType;   // 融合策略（OWL/RDF等）

    private String conflictDetails; // 冲突详情（JSON存储）

    private Integer newNodeCount;  // 新增节点数
    private Integer conflictCount; // 冲突总数

    private LocalDateTime createTime;
}
