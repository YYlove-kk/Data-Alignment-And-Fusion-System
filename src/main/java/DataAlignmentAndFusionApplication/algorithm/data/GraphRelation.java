package DataAlignmentAndFusionApplication.algorithm.data;

import lombok.AllArgsConstructor;
import lombok.Data;

import java.util.Map;

@Data
@AllArgsConstructor
public class GraphRelation {
    private String id;
    private String sourceNodeId;
    private String targetNodeId;
    private String type;
    private Map<String, Object> properties;
}
