package DataAlignmentAndFusionApplication.algorithm.strategy.impl;

import DataAlignmentAndFusionApplication.algorithm.Data.AlignmentData;
import DataAlignmentAndFusionApplication.algorithm.strategy.AlgorithmStrategy;
import org.springframework.stereotype.Component;

import java.util.Map;

@Component
public class DtwStrategy implements AlgorithmStrategy {
    @Override
    public AlignmentData execute(String algorithmType, String modelType, Map<String, Object> inputData) {
        // 实现DTW算法逻辑
        return null;
    }
}