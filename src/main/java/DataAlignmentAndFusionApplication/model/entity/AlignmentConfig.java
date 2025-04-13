package DataAlignmentAndFusionApplication.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.time.LocalDateTime;

@Data
@TableName("alignment_config")
public class AlignmentConfig {
    @TableId(type = IdType.AUTO)

    private Long id;

    /**
     * 用户Id
     */
    private Long userId;

    /**
     * 算法类型
     */
    private String algorithmType;

    /**
     * 时间粒度
     */
    private String timeGranularity;

    /**
     * 语义模型
     */
    private String semanticModel;

    /**
     * 相似度阈值
     */
    private Double similarityThreshold;

    private LocalDateTime createTime;
}
