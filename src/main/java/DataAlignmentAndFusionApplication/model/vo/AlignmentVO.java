package DataAlignmentAndFusionApplication.model.vo;

import lombok.Data;


@Data
public class AlignmentVO {
    private String PatientIds;
    private String coverage; // 相似度大于阈值的对的数量
    private Double accuracy; // 相似度大于阈值的比例
    private String alignmentMatrix;
}
