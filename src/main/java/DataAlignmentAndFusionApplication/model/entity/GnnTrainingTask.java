package DataAlignmentAndFusionApplication.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;

import java.time.LocalDateTime;
import java.util.Date;
import lombok.Data;

/**
 * 
 * @TableName gnn_training_task
 */
@TableName(value ="gnn_training_task")
@Data
public class GnnTrainingTask {
    /**
     * 
     */
    @TableId(type = IdType.AUTO)
    private Long id;

    private String taskName;
    private String status; // e.g., PENDING, RUNNING, SUCCESS, FAILED
    private String resultPath; // 模型权重保存路径
    private String log; // 错误信息或日志
    private LocalDateTime createTime;
    private LocalDateTime updateTime;
}