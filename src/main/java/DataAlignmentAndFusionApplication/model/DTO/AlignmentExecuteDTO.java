package DataAlignmentAndFusionApplication.model.DTO;

import jakarta.validation.constraints.NotNull;
import lombok.Data;

import java.util.Map;

@Data
public class AlignmentExecuteDTO {
    @NotNull(message = "用户ID不能为空")
    private Long userId;

    @NotNull(message = "输入数据不能为空")
    private Map<String, Object> inputData;  // 待对齐的原始数据
}