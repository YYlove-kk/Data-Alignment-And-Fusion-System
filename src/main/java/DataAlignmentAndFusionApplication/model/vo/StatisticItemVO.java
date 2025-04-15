package DataAlignmentAndFusionApplication.model.vo;

import lombok.Data;

@Data
public class StatisticItemVO {
    private String key;           // 如"recovery_rate"
    private String label;         // 如"康复率"
    private String value;         // 如"72.5%"
    private String conclusion;    // 如"高于平均水平(65%)"
}
