package DataAlignmentAndFusionApplication.model.DTO;

import lombok.Data;
import org.springframework.web.multipart.MultipartFile;

@Data
public class FileUploadDTO {

    private String dataSourceName;

    private String modalityType;

    private String institution;

    /**
     * Spring接收上传文件
     */
    private MultipartFile file;
}