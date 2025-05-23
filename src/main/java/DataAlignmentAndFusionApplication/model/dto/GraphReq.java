package DataAlignmentAndFusionApplication.model.dto;

import DataAlignmentAndFusionApplication.common.enums.BuildMode;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

import java.util.List;

@Data
public class GraphReq {
    private List<String> sourceIds;
    private int graphTag;
    @JsonProperty("mode")
    private int mode;
}
