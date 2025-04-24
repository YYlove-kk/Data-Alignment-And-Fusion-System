package DataAlignmentAndFusionApplication.common;

import lombok.Data;

@Data
public class UploadResp {
    // 原始文件路径
    private String rawPath;
    // 清洗后文件路径
    private String cleanPath;
    // 处理状态
    private String status;

    // 构造函数
    public UploadResp(String rawPath, String cleanPath, String status) {
        this.rawPath = rawPath;
        this.cleanPath = cleanPath;
        this.status = status;
    }
}