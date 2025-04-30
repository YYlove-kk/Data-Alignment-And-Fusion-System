package DataAlignmentAndFusionApplication.model.dto;

import DataAlignmentAndFusionApplication.model.enums.BuildMode;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

@Data
public class GraphReq {
    private String patientId;
    private int graphTag;
    @JsonProperty("mode")
    private BuildMode mode = BuildMode.MULTI;
}
