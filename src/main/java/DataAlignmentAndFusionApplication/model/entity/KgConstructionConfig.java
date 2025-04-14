package DataAlignmentAndFusionApplication.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.time.LocalDateTime;
// 图谱构建配置
@Data
@TableName("kg_construction_config")
public class KgConstructionConfig {
    @TableId(type = IdType.AUTO)
    private Long id;
    private Long userId;
    private String datasetId;        // 对齐结果数据集ID
    private String constructionMode; // single/multi
    private String fusionStrategy;   // OWL/RDF等
    private LocalDateTime createTime;
}
