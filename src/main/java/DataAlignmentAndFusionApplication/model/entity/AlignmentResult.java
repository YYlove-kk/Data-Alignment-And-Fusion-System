package DataAlignmentAndFusionApplication.model.entity;


import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import java.util.Date;
import java.util.List;

import lombok.Data;

/**
 *
 * @TableName alignment_result
 */
@TableName("alignment_result")
@Data
public class AlignmentResult {

    private Long id;  // 主键字段

    @TableField("alignment_matrix")
    private String alignmentMatrix;  // 存储对齐矩阵的 JSON 字符串

    @TableField("semantic_accuracy")
    private double semanticAccuracy;  // 语义准确率

    @TableField("alignment_coverage")
    private int alignmentCoverage;// 对齐覆盖数

    // diagonal_similarity：以 JSON 字符串形式保存
    private String diagonalSimilarity;

    // patient_ids：以 JSON 字符串形式保存
    private String sourceIds;

    private List<String> filenamePairs;
}