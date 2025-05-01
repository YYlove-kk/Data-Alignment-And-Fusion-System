package DataAlignmentAndFusionApplication.model.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serializable;
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class UploadMessage implements Serializable {

    private String rawPath;
    private String schemaRegistryPath;
    private String reportDir;
    private String cleanDir;
    private String cleanPath;
    private String outputDir;
    private String outputPath;
    private String fileName;
    private String sourceId;
    private String taskId;
    private String status;
    private String modalityType;
    private String institution;


}
