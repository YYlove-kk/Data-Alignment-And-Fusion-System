package DataAlignmentAndFusionApplication.model.DTO;

import jakarta.validation.constraints.NotBlank;
import lombok.Data;

@Data
public class ConstructionDTO {
    @NotBlank(message = "数据集ID不能为空")
    private String datasetId;

    @NotBlank(message = "构建模式不能为空")
    private String mode; // "single" 或 "multi"

    private Boolean autoFuse = false;
}
