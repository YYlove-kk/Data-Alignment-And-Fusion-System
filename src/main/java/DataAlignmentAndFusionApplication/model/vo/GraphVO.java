package DataAlignmentAndFusionApplication.model.vo;

import lombok.Data;

import java.time.LocalDateTime;

@Data
public class GraphVO {
    private String graphId;
    private String visualizationUrl;
    private LocalDateTime createTime;
}
