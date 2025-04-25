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
@TableName(value = "upload_record")
@Data
public class UploadRecord {
    /**
     * ID 主键
     */
    @TableId(type = IdType.AUTO)
    private Integer id;

    /**
     * 任务ID，唯一标识任务
     */
    private String taskId;

    /**
     * 原文件路径
     */
    private String rawPath;

    /**
     * 清洗后文件路径
     */
    private String cleanPath;

    /**
     * 模式注册路径
     */
    private String schemaRegistryPath;

    /**
     * 清洗报告路径
     */
    private String reportDir;

    /**
     * 清洗结果输出路径
     */
    private String outputDir;

    /**
     * 处理时间
     */
    private Date processTime;

    /**
     * 任务状态
     */
    private String status;
}
