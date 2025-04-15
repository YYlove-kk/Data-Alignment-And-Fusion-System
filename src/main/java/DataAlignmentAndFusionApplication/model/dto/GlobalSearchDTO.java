package DataAlignmentAndFusionApplication.model.dto;

import jakarta.validation.constraints.NotBlank;
import lombok.Data;

@Data
public class GlobalSearchDTO {
    @NotBlank
    private String keyword;                   // 支持患者ID/疾病名等

//    @NotNull
//    private SearchScope scope;                // 枚举: ALL/UPLOAD/ALIGNMENT等

    private Boolean exactMatch = false;       // 是否精确匹配
}