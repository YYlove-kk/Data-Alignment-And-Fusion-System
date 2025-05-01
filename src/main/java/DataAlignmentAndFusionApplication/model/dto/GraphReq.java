package DataAlignmentAndFusionApplication.model.dto;

import DataAlignmentAndFusionApplication.model.enums.BuildMode;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

import java.util.List;

@Data
public class GraphReq {
    private List<String> sourceIds;
    private int graphTag;
    @JsonProperty("mode")
    private BuildMode mode = BuildMode.MULTI;
}
