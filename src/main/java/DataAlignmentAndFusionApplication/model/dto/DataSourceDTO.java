package DataAlignmentAndFusionApplication.model.dto;

import lombok.Data;

@Data
public class DataSourceDTO {

    private String name;

    private String modalityType;

    private String institution;

    private String description;

    private String filePath;
}