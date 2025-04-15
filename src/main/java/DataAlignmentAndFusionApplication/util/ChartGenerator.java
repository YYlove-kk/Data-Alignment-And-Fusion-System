package DataAlignmentAndFusionApplication.util;

import DataAlignmentAndFusionApplication.model.vo.ChartDataVO;
import org.springframework.stereotype.Component;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Component
public class ChartGenerator {
    public ChartDataVO generateBarChart(Map<String, Double> data) {
        Map<String, Object> config = new HashMap<>();
        config.put("xAxis", Map.of("data", data.keySet()));
        config.put("series", List.of(
                Map.of("name", "数值", "data", data.values())
        ));
        ChartDataVO chartDataVO = new ChartDataVO();
        chartDataVO.setType("bar");
        chartDataVO.setConfig(config);
        return chartDataVO;
    }

    public ChartDataVO generatePieChart(Map<String, Double> data) {
        // 实现类似逻辑...
        return null;
    }
}