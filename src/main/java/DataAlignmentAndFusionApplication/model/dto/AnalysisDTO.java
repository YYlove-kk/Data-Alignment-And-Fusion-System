package DataAlignmentAndFusionApplication.model.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotEmpty;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Pattern;
import lombok.Data;

import java.util.HashSet;
import java.util.Map;
import java.util.Set;

@Data
public class AnalysisDTO {
    @NotNull(message = "用户ID不能为空")
    private Long userId;          // 从请求头自动注入，不暴露给前端

    @NotBlank(message = "分析类型不能为空")
    @Pattern(regexp = "patient|disease", message = "分析类型必须是patient或disease")
    private String analysisType;  // 患者分析(patient)或疾病分析(disease)

    @NotBlank(message = "目标ID不能为空")
    private String targetId;      // 患者ID或疾病名称

    @NotEmpty(message = "数据源不能为空")
    private Set<String> dataSources = new HashSet<>(); // 数据源集合

    @NotBlank(message = "分析模式不能为空")
    private String mode;          // basic(基础分析)/advanced(高级分析)


    private Map<String, Object> customParams; // 自定义参数

    // 辅助方法：验证疾病分析时的参数
    public boolean isDiseaseAnalysis() {
        return "disease".equalsIgnoreCase(this.analysisType);
    }
}