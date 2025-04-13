package DataAlignmentAndFusionApplication.algorithm.strategy;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

@Service
public class AlgorithmFactory {
    private final Map<String, AlgorithmStrategy> strategyMap;

    @Autowired
    public AlgorithmFactory(List<AlgorithmStrategy> strategies) {
        this.strategyMap = strategies.stream()
                .collect(Collectors.toMap(
                        s -> s.getClass().getSimpleName().replace("Strategy", "").toLowerCase(),
                        Function.identity()
                ));
    }

    public AlgorithmStrategy getStrategy(String algorithmType) {
        // 获取策略实现
        return strategyMap.get(algorithmType);
    }

}