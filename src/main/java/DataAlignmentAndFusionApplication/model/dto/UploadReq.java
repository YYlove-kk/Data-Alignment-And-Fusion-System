package DataAlignmentAndFusionApplication.model.dto;

import lombok.Data;
import org.springframework.web.multipart.MultipartFile;

@Data
public class UploadReq {
    private String sourceId;

    private String modalityType;

    private String institution;

    /**
     * Spring接收上传文件
     */
    private MultipartFile file;
}