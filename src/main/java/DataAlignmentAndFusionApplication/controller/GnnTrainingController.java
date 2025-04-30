package DataAlignmentAndFusionApplication.controller;

import DataAlignmentAndFusionApplication.model.entity.GnnTrainingTask;
import DataAlignmentAndFusionApplication.service.GnnTrainingService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/gnn")
public class GnnTrainingController {

    @Autowired
    private GnnTrainingService gnnTrainingService;

    @PostMapping("/train")
    public ResponseEntity<Long> startTraining(@RequestParam String taskName) {
        Long taskId = gnnTrainingService.startTraining(taskName);
        return ResponseEntity.ok(taskId);
    }

    @GetMapping("/task/{id}")
    public ResponseEntity<GnnTrainingTask> getTask(@PathVariable Long id) {
        return ResponseEntity.ok(gnnTrainingService.getTaskById(id));
    }

    @GetMapping("/status")
    public ResponseEntity<GnnTrainingTask> getStatus(@RequestParam String taskId) {
        GnnTrainingTask task = gnnTrainingService.getTaskByTaskId(taskId);
        return task != null ? ResponseEntity.ok(task) : ResponseEntity.notFound().build();
    }

}
