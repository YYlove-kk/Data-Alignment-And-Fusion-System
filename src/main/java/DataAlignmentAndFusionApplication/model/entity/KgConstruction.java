package DataAlignmentAndFusionApplication.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.time.LocalDateTime;
import java.util.Map;
//构建过程
@Data
@TableName("kg_construction")
public class KgConstruction {
    @TableId(type = IdType.AUTO)
    private Long id;

    private Long userId;
    private String datasetId;
    private String constructionMode; // single/multi

//    @Column(typeHandler = JsonTypeHandler.class)
    private Map<String, Object> config; // 策略配置

//    @Column(typeHandler = JsonTypeHandler.class)
    private Map<String, Object> result; // 构建结果

    private String graphId;         // 生成的图谱ID
    private String recordType;      // CONFIG/RECORD（类型标识）
    private LocalDateTime createTime;
}