package DataAlignmentAndFusionApplication.service;

import DataAlignmentAndFusionApplication.model.entity.GnnTrainingTask;
import com.baomidou.mybatisplus.extension.service.IService;

/**
 * @author 29857
 * @description 针对表【gnn_training_task】的数据库操作Service
 * @createDate 2025-05-01 01:02:40
 */
public interface GnnTrainingService extends IService<GnnTrainingTask> {

    Long startTraining(String taskName);

    GnnTrainingTask getTaskById(Long id);

    GnnTrainingTask getTaskByTaskId(String taskId);

}
