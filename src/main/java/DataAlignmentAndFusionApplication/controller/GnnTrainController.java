package DataAlignmentAndFusionApplication.controller;



import DataAlignmentAndFusionApplication.model.entity.TrainRecord;
import DataAlignmentAndFusionApplication.service.TrainRecordService;
import DataAlignmentAndFusionApplication.util.Result;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/gnn")
public class GnnTrainController {

    @Autowired
    private TrainRecordService trainRecordService;

    @PostMapping("/train")
    public Result<String> startTraining() {
        return trainRecordService.startTraining();
    }

    @GetMapping("/record")
    public List<TrainRecord> getRecord() {
        return trainRecordService.getAll();
    }

    @GetMapping("/models")
    public List<String> getModelNames() {
        return trainRecordService.getModelNames();
    }

}
