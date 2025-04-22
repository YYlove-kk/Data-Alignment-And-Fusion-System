package DataAlignmentAndFusionApplication.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import java.util.Date;
import lombok.Data;

/**
 * 数据源管理表
 * @TableName data_source
 */
@TableName(value ="data_source")
@Data
public class DataSource {
    /**
     * 
     */
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

    /**
     * 
     */
    private Date createTime;

    /**
     * 
     */
    private Date updateTime;


}