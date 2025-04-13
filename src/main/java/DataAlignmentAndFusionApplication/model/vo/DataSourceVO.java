package DataAlignmentAndFusionApplication.model.vo;

import DataAlignmentAndFusionApplication.model.entity.DataSource;
import com.fasterxml.jackson.annotation.JsonFormat;
import lombok.Data;

import java.time.LocalDateTime;

/**
 * 用于前端展示数据源详情
 */
@Data
public class DataSourceVO {

    private Long id;
    private String name;
    private String modalityType;
    private String institution;
    private String description;
    private String creator;
    private Integer fileCount;

    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;

    public static DataSourceVO fromEntity(DataSource entity) {
        DataSourceVO vo = new DataSourceVO();
        vo.setId(entity.getId());
        vo.setName(entity.getName());
        vo.setModalityType(entity.getModalityType());
        vo.setInstitution(entity.getInstitution());
        vo.setDescription(entity.getDescription());
        vo.setCreator(entity.getCreator());
        vo.setCreateTime(entity.getCreateTime());

        return vo;
    }
}
