package DataAlignmentAndFusionApplication.model.entity;

import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import java.util.Date;
import lombok.Data;

/**
 * 
 * @TableName analysis_record
 */
@TableName(value ="analysis_record")
@Data
public class AnalysisRecord {
    /**
     * 
     */
    @TableId
    private Long id;

    /**
     * 
     */
    private Long userId;

    /**
     * 
     */
    private String analysisType;

    /**
     * 
     */
    private Object dataSources;

    /**
     * 
     */
    private String analysisMode;

    /**
     * 
     */
    private Object parameters;

    /**
     * 
     */
    private Object resultStats;

    /**
     * 
     */
    private Date createTime;
}