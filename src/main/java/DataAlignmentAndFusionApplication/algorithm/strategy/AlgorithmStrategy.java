package DataAlignmentAndFusionApplication.algorithm.strategy;

import DataAlignmentAndFusionApplication.algorithm.data.AlignmentData;


import java.util.Map;


public interface AlgorithmStrategy {
    /**
     * 执行对齐算法
     * @param algorithmType 算法类型（如DTW/ED）
     * @param modelType 语义模型类型（如BERT）
     * @param inputData 输入数据
     * @return 对齐结果数据
     */
    AlignmentData execute(String algorithmType, String modelType, Map<String, Object> inputData);
}
