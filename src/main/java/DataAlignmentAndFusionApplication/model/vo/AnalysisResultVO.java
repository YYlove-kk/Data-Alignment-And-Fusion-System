package DataAlignmentAndFusionApplication.model.vo;

import lombok.Data;

import java.util.List;

@Data
public class AnalysisResultVO {
    private String analysisId;
    private String title;         // 如"糖尿病患者年龄分布"
    private List<StatisticItemVO> stats; // 统计项列表
    private ChartDataVO chartData; // 图表数据
}
