package DataAlignmentAndFusionApplication.algorithm.strategy;

import DataAlignmentAndFusionApplication.model.vo.AnalysisResultVO;

public interface AnalysisStrategy {
    AnalysisResultVO analyze(AnalysisRequest request);
}