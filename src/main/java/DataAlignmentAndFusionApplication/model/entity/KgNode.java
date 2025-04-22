package DataAlignmentAndFusionApplication.model.entity;

import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import java.util.Date;
import lombok.Data;

/**
 * 
 * @TableName kg_node
 */
@TableName(value ="kg_node")
@Data
public class KgNode {
    /**
     * 
     */
    @TableId
    private String nodeId;

    /**
     * 
     */
    private String graphId;

    /**
     * 
     */
    private String nodeType;

    /**
     * 
     */
    private String name;

    /**
     * 
     */
    private String dataSource;

    /**
     * 
     */
    private Integer version;

    /**
     * 
     */
    private Object extendedProps;

    /**
     * 
     */
    private Date createTime;

    /**
     * 
     */
    private Date updateTime;

}