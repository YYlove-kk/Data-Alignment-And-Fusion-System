package DataAlignmentAndFusionApplication.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import java.util.Date;
import java.util.List;

import com.baomidou.mybatisplus.extension.handlers.JacksonTypeHandler;
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

    private String sourceId;

    private String fileName;

    /**
     * 原文件路径
     */
    private String rawDir;

    /**
     * 清洗结果输出路径
     */
    private String cleanDir;

    private String cleanPath;
    /**
     * 嵌入结果输出路径
     */
    private String outputDir;

    private String outputPath;

    @TableField(typeHandler = JacksonTypeHandler.class)
    private List<String> singleEmbedNpy;

    /**
     * 模式注册路径
     */
    private String schemaRegistryPath;

    /**
     * 清洗报告路径
     */
    private String reportDir;
    //模态类型
    private String modalityType;
    //机构
    private String institution;

    /**
     * 处理时间
     */
    private Date processTime;

    /**
     * 任务状态
     */
    private String status;
}
