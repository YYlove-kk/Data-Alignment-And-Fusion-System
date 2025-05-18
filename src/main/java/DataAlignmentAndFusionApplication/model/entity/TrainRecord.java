package DataAlignmentAndFusionApplication.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;

import java.time.LocalDateTime;
import java.util.Date;
import lombok.Data;

/**
 * 
 * @TableName train_record
 */
@TableName(value ="train_record")
@Data
public class TrainRecord {
    /**
     * 
     */
    @TableId(type = IdType.AUTO)
    private Long id;

    /**
     * 
     */
    private Integer epoch;

    /**
     * 
     */
    private String status;

    private String resultPath;
    /**
     * 
     */
    private LocalDateTime createTime;

    /**
     * 
     */
    private String hits1;

    /**
     * 
     */
    private String hits5;

    /**
     * 
     */
    private String hits10;

    /**
     * 
     */
    private String trainLoss;

    /**
     * 
     */
    private String testLoss;
}