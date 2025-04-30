package DataAlignmentAndFusionApplication.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import DataAlignmentAndFusionApplication.model.entity.GnnTrainingTask;
import DataAlignmentAndFusionApplication.service.GnnTrainingService;
import DataAlignmentAndFusionApplication.mapper.GnnTrainingTaskMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.time.LocalDateTime;

/**
 * @author 29857
 * @description 针对表【gnn_training_task】的数据库操作Service实现
 * @createDate 2025-05-01 01:02:40
 */
@Service
public class GnnTrainingServiceImpl extends ServiceImpl<GnnTrainingTaskMapper, GnnTrainingTask>
        implements GnnTrainingService {

    @Autowired
    private GnnTrainingTaskMapper taskMapper;

    @Override
    public Long startTraining(String taskName) {
        GnnTrainingTask task = new GnnTrainingTask();
        task.setTaskName(taskName);
        task.setStatus("PENDING");
        task.setCreateTime(LocalDateTime.now());
        task.setUpdateTime(LocalDateTime.now());
        taskMapper.insert(task);

        // 启动异步任务（建议使用线程池或 Spring @Async）
        new Thread(() -> runTraining(task)).start();

        return task.getId();
    }

    private void runTraining(GnnTrainingTask task) {
        try {
            task.setStatus("RUNNING");
            task.setUpdateTime(LocalDateTime.now());
            taskMapper.updateById(task);

            ProcessBuilder pb = new ProcessBuilder("python", "train_han.py", String.valueOf(task.getId()));
            pb.redirectErrorStream(true);
            Process process = pb.start();

            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            StringBuilder output = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                output.append(line).append("\n");
            }

            int code = process.waitFor();
            if (code == 0) {
                task.setStatus("SUCCESS");
                task.setResultPath("han_epoch99.pt");
                task.setLog(output.toString());
            } else {
                task.setStatus("FAILED");
                task.setLog(output.toString());
            }
        } catch (Exception e) {
            task.setStatus("FAILED");
            task.setLog(e.getMessage());
        } finally {
            task.setUpdateTime(LocalDateTime.now());
            taskMapper.updateById(task);
        }
    }

    @Override
    public GnnTrainingTask getTaskById(Long id) {
        return taskMapper.selectById(id);
    }

    @Override
    public GnnTrainingTask getTaskByTaskId(String taskId) {
        return taskMapper.selectOne(new QueryWrapper<GnnTrainingTask>().eq("task_id", taskId));
    }

}



