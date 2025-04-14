package DataAlignmentAndFusionApplication.model.vo;

import lombok.Data;

import java.util.List;
import java.util.Map;

@Data
public class FusionHistoryVO {
    private Long recordId;
    private String mainGraphName;
    private String subGraphName;
    private Integer newNodeCount;
    private Integer conflictCount;
    private List<Map<String, Object>> snapshot; // 解析后的JSON快照
}
