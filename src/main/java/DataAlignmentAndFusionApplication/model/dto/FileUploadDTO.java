package DataAlignmentAndFusionApplication.model.dto;

import lombok.Data;
import org.springframework.web.multipart.MultipartFile;

@Data
public class FileUploadDTO {

    private String modalityType;

    private String institution;

    private String schemaRegistryPath;

    /**
     * Spring接收上传文件
     */
    private MultipartFile file;
}