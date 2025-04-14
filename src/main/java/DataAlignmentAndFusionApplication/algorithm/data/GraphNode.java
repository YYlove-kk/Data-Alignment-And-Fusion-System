package DataAlignmentAndFusionApplication.algorithm.data;

import lombok.AllArgsConstructor;
import lombok.Data;

import java.util.Map;

@Data
@AllArgsConstructor
public class GraphNode {
    private String id;
    private String label;
    private Map<String, Object> properties;
    private String dataSource; // 来源数据集
}