package DataAlignmentAndFusionApplication.model.vo;

import lombok.Data;

@Data
public class AlignmentDetailVO {
    private String sourceData;
    private String targetData;
    private Double similarity;
    private String status;            // "高相似度"/"低相似度"
}
