package DataAlignmentAndFusionApplication.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import java.util.Date;
import lombok.Data;

/**
 * 
 * @TableName upload_record
 */
@TableName(value ="upload_record")
@Data
public class UploadRecord {
    /**
     * 
     */
    @TableId(type = IdType.AUTO)
    private Long id;

    /**
     * 
     */
    private String dataSourceName;

    /**
     * 
     */
    private String modalityType;

    /**
     * 
     */
    private String institution;

    /**
     * 
     */
    private String filePath;

    /**
     * 
     */
    private String fileName;

    /**
     * 
     */
    private Long fileSize;

    /**
     * 
     */
    private String status;

    /**
     * 
     */
    private String errorDetail;

    /**
     * 
     */
    private Date uploadTime;

}