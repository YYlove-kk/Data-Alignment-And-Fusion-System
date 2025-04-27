package DataAlignmentAndFusionApplication.model.dto;

import lombok.Data;

@Data
public class AlignmentRequest {
    private String taskId;
    private String textDir;
    private String imageDir;

}

