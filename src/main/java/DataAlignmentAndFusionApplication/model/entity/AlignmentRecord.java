package DataAlignmentAndFusionApplication.model.entity;

import lombok.Data;

@Data
public class AlignmentRecord {
    private String patientId;
    private String fileName;
    private Double semanticSimilarity;
}
