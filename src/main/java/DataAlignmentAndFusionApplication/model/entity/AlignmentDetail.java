package DataAlignmentAndFusionApplication.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

@Data
@TableName("alignment_detail")
public class AlignmentDetail {
    @TableId(type = IdType.AUTO)
    private Long id;

    /**
     * 关联结果ID
     */
    private Long resultId;

    /**
     * 源数据
     */
    private String sourceData;

    /**
     * 目标数据
     */
    private String targetData;

    /**
     * 相似度得分
     */
    private Double similarity;
}
