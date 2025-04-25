package DataAlignmentAndFusionApplication.common;

import DataAlignmentAndFusionApplication.model.enums.ResponseCode;
import lombok.Data;

@Data
public class UploadResp {
    // 原始文件路径
    private final String rawPath;
    // 清洗后文件路径
    private final String cleanPath;
    // 处理状态
    private final int status;
    // 消息信息
    private final String message;

    // 构造函数
    public UploadResp(String rawPath, String cleanPath, int status, String message) {
        this.rawPath = rawPath;
        this.cleanPath = cleanPath;
        this.status = status;
        this.message = message;
    }

    // 判断上传是否成功
    public boolean isSuccess() {
        return this.status == ResponseCode.SUCCESS.getCode();
    }

    // 创建表示上传成功的响应
    public static UploadResp createForSuccess(String rawPath, String cleanPath, String message) {
        return new UploadResp(rawPath, cleanPath, ResponseCode.SUCCESS.getCode(), message);
    }

    // 创建表示上传失败的响应
    public static UploadResp createForError(String message) {
        return new UploadResp(null, null, ResponseCode.ERROR.getCode(), message);
    }
}