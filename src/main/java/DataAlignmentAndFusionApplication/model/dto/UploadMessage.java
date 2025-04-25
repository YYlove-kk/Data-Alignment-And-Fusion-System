package DataAlignmentAndFusionApplication.model.dto;

import lombok.Data;

import java.io.Serializable;
@Data
public class UploadMessage implements Serializable {

    private String rawPath;
    private String cleanPath;
    private String schemaRegistryPath;
    private String reportDir;
    private String outputDir;
    private String fileName;
    private String taskId;
    private String status;

    // RabbitMQ 反序列化用
    public UploadMessage() {}

}
