package DataAlignmentAndFusionApplication.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

/**
 * 
 * @TableName reduce_record
 */
@TableName(value ="reduce_record")
@Data
public class ReduceRecord {
    /**
     * 
     */
    @TableId(type = IdType.AUTO)
    private Integer id;

    /**
     * 
     */
    private String sourceId;

    /**
     * 
     */
    private String status;

    /**
     * 
     */
    private String filename;

    private String patientId;
}