package DataAlignmentAndFusionApplication.model.vo;

import DataAlignmentAndFusionApplication.model.entity.UploadRecord;
import com.fasterxml.jackson.annotation.JsonFormat;
import lombok.Data;

import java.time.LocalDateTime;

/**
 * 用于前端展示上传记录列表
 */
@Data
public class UploadRecordVO {

    private Long id;
    private String dataSourceName;  // 数据源名称（可关联到DataSource）
    private String modalityType;    // 模态类型（文本/图像/时序等）
    private String institution;     // 机构名称
    private String fileName;        // 原始文件名
    private String fileSize;        // 格式化后的文件大小（如"10.5 MB"）
    private String status;          // 状态（处理中/成功/失败）

    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime uploadTime;

    // 从实体类转换的静态方法
    public static UploadRecordVO fromEntity(UploadRecord entity) {
        UploadRecordVO vo = new UploadRecordVO();
        vo.setId(entity.getId());
        vo.setDataSourceName(entity.getDataSourceName());
        vo.setModalityType(entity.getModalityType());
        vo.setInstitution(entity.getInstitution());
        vo.setFileName(entity.getFileName());
        vo.setFileSize(formatFileSize(entity.getFileSize())); // 格式化文件大小
        vo.setStatus(entity.getStatus());
        vo.setUploadTime(entity.getUploadTime());
        return vo;
    }

    // 格式化文件大小（字节 → KB/MB/GB）
    private static String formatFileSize(Long size) {
        if (size == null) return "0 B";
        if (size < 1024) return size + " B";
        if (size < 1024 * 1024) return String.format("%.1f KB", size / 1024.0);
        if (size < 1024 * 1024 * 1024) return String.format("%.1f MB", size / (1024.0 * 1024));
        return String.format("%.1f GB", size / (1024.0 * 1024 * 1024));
    }
}
