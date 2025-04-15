package DataAlignmentAndFusionApplication.model.dto;


import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

//融合请求传输对象
@Data
public class FusionDTO {
    @NotNull(message = "主图谱ID不能为空")
    private Long mainGraphId;

    @NotNull(message = "子图谱ID不能为空")
    private Long subGraphId;

    @NotBlank(message = "融合策略不能为空")
    private String strategy; // OWL/RDF等

}