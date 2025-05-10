package DataAlignmentAndFusionApplication.service;

import DataAlignmentAndFusionApplication.model.entity.TrainRecord;
import DataAlignmentAndFusionApplication.util.Result;
import com.baomidou.mybatisplus.extension.service.IService;

import java.util.List;

/**
* @author 29857
* @description 针对表【train_record】的数据库操作Service
* @createDate 2025-05-07 23:16:58
*/
public interface TrainRecordService extends IService<TrainRecord> {
    Result<String> startTraining();

    List<TrainRecord> getAll();

    List<String> getModelNames();
}
