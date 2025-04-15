package DataAlignmentAndFusionApplication.model.dto;

import jakarta.validation.constraints.DecimalMax;
import jakarta.validation.constraints.DecimalMin;
import jakarta.validation.constraints.NotBlank;

public class AlignmentConfigDTO {
    @NotBlank(message = "算法类型不能为空")
    private String algorithmType;

    @NotBlank(message = "时间粒度不能为空")
    private String timeGranularity;

    @NotBlank(message = "语义模型不能为空")
    private String semanticModel;

    @DecimalMin(value = "0.0", message = "阈值必须≥0")
    @DecimalMax(value = "1.0", message = "阈值必须≤1")
    private Double similarityThreshold;
}
