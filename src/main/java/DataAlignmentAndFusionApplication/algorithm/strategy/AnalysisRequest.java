package DataAlignmentAndFusionApplication.algorithm.strategy;

import lombok.Data;

import java.util.Map;
import java.util.Set;

@Data
public class AnalysisRequest {
    private String targetId;      // 患者ID或疾病名称
    private Set<String> dataSources;
    private String mode;          // basic/advanced
    private Map<String, Object> customParams;
}