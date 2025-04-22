package DataAlignmentAndFusionApplication.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import java.util.Date;
import lombok.Data;

/**
 * 
 * @TableName alignment_config
 */
@TableName(value ="alignment_config")
@Data
public class AlignmentConfig {
    /**
     * 
     */
    @TableId(type = IdType.AUTO)
    private Long id;

    /**
     * 用户ID
     */
    private String userId;

    /**
     * 时序对齐算法
     */
    private String timeSeriesAlgo;

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

    /**
     * 
     */
    private Date createTime;


}