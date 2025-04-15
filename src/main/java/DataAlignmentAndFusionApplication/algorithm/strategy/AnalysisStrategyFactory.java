package DataAlignmentAndFusionApplication.algorithm.strategy;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

@Service
public class AnalysisStrategyFactory {
    private final Map<String, AnalysisStrategy> strategyMap;

    @Autowired
    public AnalysisStrategyFactory(List<AnalysisStrategy> strategies) {
        this.strategyMap = strategies.stream()
                .collect(Collectors.toMap(
                        s -> s.getClass().getSimpleName()
                                .replace("AnalysisStrategy", "").toLowerCase(),
                        Function.identity()
                ));
    }

    public AnalysisStrategy getStrategy(String type) {
        AnalysisStrategy strategy = strategyMap.get(type.toLowerCase());
        if (strategy == null) {
            throw new IllegalArgumentException("未知分析类型: " + type);
        }
        return strategy;
    }
}
