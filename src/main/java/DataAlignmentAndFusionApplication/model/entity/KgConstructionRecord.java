package DataAlignmentAndFusionApplication.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.time.LocalDateTime;

@Data
@TableName("kg_construction_record")
public class KgConstructionRecord {
    @TableId(type = IdType.AUTO)
    private Long id;

    private Long userId;          // 用户ID
    private String datasetId;     // 使用的数据集ID
    private String constructionMode; // 构建模式（single/multi）

    private String graphConfig;   // 构建配置（JSON存储）

    private String graphId;       // 生成的图谱ID（关联图数据库）
    private LocalDateTime createTime;
}
