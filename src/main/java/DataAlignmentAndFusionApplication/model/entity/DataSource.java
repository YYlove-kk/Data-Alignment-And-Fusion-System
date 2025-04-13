package DataAlignmentAndFusionApplication.model.entity;
import com.baomidou.mybatisplus.annotation.TableName;
import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("data_source")
public class DataSource {
    @TableId(type = IdType.AUTO)
    private Long id;

    /**
     * 数据源名称
     */
    private String name;

    /**
     * 模态类型
     */
    private String modalityType;

    /**
     * 机构名称
     */
    private String institution;

    /**
     * 数据源描述
     */
    private String description;

    /**
     * 创建人
     */
    private String creator;

    private LocalDateTime createTime;
    private LocalDateTime updateTime;
}
