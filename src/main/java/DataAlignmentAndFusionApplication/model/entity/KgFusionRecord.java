package DataAlignmentAndFusionApplication.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import java.util.Date;
import lombok.Data;

/**
 * 
 * @TableName kg_fusion_record
 */
@TableName(value ="kg_fusion_record")
@Data
public class KgFusionRecord {
    /**
     * 
     */
    @TableId(type = IdType.AUTO)
    private Long id;

    /**
     * 
     */
    private Long mainGraphId;

    /**
     * 
     */
    private Long subGraphId;

    /**
     * 
     */
    private String strategyType;

    /**
     * 
     */
    private Object conflictDetails;

    /**
     * 
     */
    private Integer newNodeCount;

    /**
     * 
     */
    private Integer conflictCount;

    /**
     * 
     */
    private Date createTime;


}