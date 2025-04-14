package DataAlignmentAndFusionApplication.algorithm.data;

import lombok.Data;

import java.util.List;

@Data
public class AlignmentData {
    private Double avgSimilarity;
    private List<AlignmentDetailData> details;

    @Data
    public static class AlignmentDetailData {
        private String sourceData;
        private String targetData;
        private Double similarity;
    }
}
