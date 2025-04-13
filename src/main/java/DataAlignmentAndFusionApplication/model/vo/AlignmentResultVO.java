package DataAlignmentAndFusionApplication.model.vo;

import lombok.Data;

import java.util.List;

@Data
public class AlignmentResultVO {
    private Long resultId;
    private Double avgSimilarity;
    private String chartData;          // Base64图片或URL
    private String createTime;         // 格式化时间
    private List<AlignmentDetailVO> details;
}
