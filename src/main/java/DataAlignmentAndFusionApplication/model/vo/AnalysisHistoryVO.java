package DataAlignmentAndFusionApplication.model.vo;

import lombok.Data;

import java.time.LocalDateTime;

@Data
public class AnalysisHistoryVO {
    private Long recordId;          // 分析记录ID
    private String analysisType;    // 分析类型（patient/disease）
    private String targetDisplay;   // 目标显示名称（患者姓名/疾病名称）
    private String dataSources;     // 数据来源简写（如"3家医院"）
    private String mode;            // 分析模式（基础/高级）
    private LocalDateTime createTime; // 分析时间
    private String summary;         // 核心结论摘要


}
