package DataAlignmentAndFusionApplication.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;

import java.time.LocalDateTime;
import lombok.Data;

/**
 * 
 * @TableName build_record
 */
@TableName(value ="build_record")
@Data
public class BuildRecord {
    /**
     * 
     */
    @TableId(type = IdType.AUTO)
    private Long id;

    /**
     * 
     */
    private String sourceId;

    /**
     * 
     */
    private Integer graphTag;

    /**
     * 
     */
    private Integer mode;

    /**
     * 
     */
    private LocalDateTime createTime;
}