package DataAlignmentAndFusionApplication.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

/**
 * 
 * @TableName alignment_detail
 */
@TableName(value ="alignment_detail")
@Data
public class AlignmentDetail {
    /**
     * 
     */
    @TableId(type = IdType.AUTO)
    private Long id;

    /**
     * 
     */
    private Long resultId;

    /**
     * 
     */
    private String sourceData;

    /**
     * 
     */
    private String targetData;

    /**
     * 
     */
    private Double similarity;
}