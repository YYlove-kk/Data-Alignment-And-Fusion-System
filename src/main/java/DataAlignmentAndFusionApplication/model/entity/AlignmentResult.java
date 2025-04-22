package DataAlignmentAndFusionApplication.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import java.util.Date;
import lombok.Data;

/**
 * 
 * @TableName alignment_result
 */
@TableName(value ="alignment_result")
@Data
public class AlignmentResult {
    /**
     * 
     */
    @TableId(type = IdType.AUTO)
    private Long id;

    /**
     * 关联的配置ID
     */
    private Long configId;

    /**
     * 对齐结果JSON
     */
    private String resultJson;

    /**
     * 图表数据JSON
     */
    private String chartData;

    /**
     * 平均相似度
     */
    private Double avgSimilarity;

    /**
     * 任务状态
     */
    private String status;

    /**
     * 
     */
    private Date createTime;

}