package DataAlignmentAndFusionApplication.algorithm.strategy.impl;

import DataAlignmentAndFusionApplication.algorithm.strategy.AnalysisRequest;
import DataAlignmentAndFusionApplication.algorithm.strategy.AnalysisStrategy;
import DataAlignmentAndFusionApplication.model.vo.AnalysisResultVO;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;
// 患者个体分析策略
@Component
@ConditionalOnProperty(name = "analysis.type", havingValue = "patient")
public class PatientAnalysisStrategy implements AnalysisStrategy {
    @Override
    public AnalysisResultVO analyze(AnalysisRequest request) {
        // 实现患者数据统计逻辑
//        return buildPatientResult(request.getTargetId());
        return null;
    }



}