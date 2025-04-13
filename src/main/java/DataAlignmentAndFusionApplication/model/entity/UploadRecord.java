package DataAlignmentAndFusionApplication.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.time.LocalDateTime;

@Data
@TableName("upload_record")
public class UploadRecord {
    @TableId(type = IdType.AUTO)
    private Long id;
    /**
     * 数据源名称
     */
    private String dataSourceName;

    /**
     * 模态类型
     */
    private String modalityType;

    /**
     * 机构名称
     */
    private String institution;

    /**
     * 文件储存路径
     */
    private String filePath;

    /**
     * 文件名
     */
    private String fileName;

    /**
     * 文件大小
     */
    private Long fileSize;

    /**
     * 上传状态
     */
    private String status;

    /**
     * 错误详情
     */
    private String errorDetail;

    /**
     * 上传时间
     */
    private LocalDateTime uploadTime;
}
