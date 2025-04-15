package DataAlignmentAndFusionApplication.model.vo;

import lombok.Data;

import java.util.Map;

@Data
public class ChartDataVO {
    private String type;          // bar/pie/line
    private Map<String, Object> config; // Echarts配置
}