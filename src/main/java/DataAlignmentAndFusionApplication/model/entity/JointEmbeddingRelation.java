package DataAlignmentAndFusionApplication.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import java.util.Date;
import lombok.Data;

/**
 * 
 * @TableName joint_embedding_relation
 */
@TableName(value ="joint_embedding_relation")
@Data
public class JointEmbeddingRelation {
    /**
     * 
     */
    @TableId(type = IdType.AUTO)
    private Integer id;

    /**
     * 
     */
    private String patientId;

    /**
     * 
     */
    private String textFile;

    /**
     * 
     */
    private String imageFile;

    /**
     * 
     */
    private Date createdAt;
}