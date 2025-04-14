package DataAlignmentAndFusionApplication.model.vo;

import lombok.Data;

@Data
public class ConflictVO {
    private String node1;
    private String node2;
    private String type;
    private String resolution;
}
