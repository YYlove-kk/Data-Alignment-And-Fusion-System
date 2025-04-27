package DataAlignmentAndFusionApplication.listener;

import DataAlignmentAndFusionApplication.config.AppConfig;
import DataAlignmentAndFusionApplication.mapper.module.UploadRecordMapper;
import DataAlignmentAndFusionApplication.model.dto.UploadMessage;
import DataAlignmentAndFusionApplication.service.UploadRecordService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.amqp.rabbit.annotation.RabbitListener;
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

@Slf4j
@Component
public class UploadMessageListener {


    @Autowired
    private RabbitTemplate rabbitTemplate;

    @Autowired
    private UploadRecordMapper uploadRecordMapper;

    @Autowired
    private UploadRecordService uploadRecordService;

    @Value("${mq.cleaning-to-database}")
    private String cleaningToDatabaseQueue;

    @Value("${mq.embedding-task}")
    private String embeddingTaskQueue;

    private final AppConfig appConfig;

    public UploadMessageListener(AppConfig appConfig) {
        this.appConfig = appConfig;
    }

    @RabbitListener(queues = "${mq.upload-to-cleaning}")
    public void handleUploadMessage(UploadMessage message) {
        try {
            // 调用 Python
            String interpreter = appConfig.getInterpreterPath();

            String fileName = message.getFileName();
            String scriptPath;

            if (fileName.endsWith(".zip")) {
                scriptPath = appConfig.getImageEmbedPath(); // 针对 DICOM 图像
                String outputPath = runPythonScript(interpreter, scriptPath,
                        "--patient_folder", message.getRawPath(),
                        "--output_dir", message.getOutputDir()+"image/");

                if (outputPath.equals("image/")) {
                    log.error("Python嵌入失败: 文件={}", message.getRawPath());
                    return;
                }
                // 成功处理，设置嵌入后的路径
                message.setCleanPath(message.getRawPath());
                message.setOutputPath(outputPath.trim());  // trim一下保险
                uploadRecordService.updatePaths(message.getTaskId(),message.getCleanPath(),message.getOutputPath());
                rabbitTemplate.convertAndSend(cleaningToDatabaseQueue, message);
            } else {
                // 先清洗
                String ingestPath = appConfig.getDataIngestPath();
                String cleanPath = runPythonScript(interpreter, ingestPath,
                        "--file_path", message.getRawPath(),
                        "--clean_dir", message.getCleanDir(),
                        "--registry_path", message.getSchemaRegistryPath(),
                        "--report_dir", message.getReportDir());

                if (cleanPath == null) {
                    log.error("Python清洗失败: 文件={}", message.getRawPath());
                    return;
                }
                // 清洗后的路径
                message.setCleanPath(cleanPath.trim());  // trim一下保险
                // 数据嵌入
                scriptPath = appConfig.getTextTimeEmbedPath();

                String outputPath = runPythonScript(interpreter, scriptPath,
                        "--file_path", message.getCleanPath(),
                        "--output_dir", message.getOutputDir()+"text/");

                if (outputPath.equals("text/")) {
                    log.error("Python嵌入失败: 文件={}", message.getRawPath());
                    return;
                }

                message.setOutputPath(outputPath.trim());
                uploadRecordService.updatePaths(message.getTaskId(),message.getCleanPath(),message.getOutputPath());
                rabbitTemplate.convertAndSend(cleaningToDatabaseQueue, message);

            }

        } catch (Exception e) {
            log.error("清洗处理异常: 文件={}", message.getRawPath(), e);
        }
    }

    @RabbitListener(queues = "${mq.cleaning-to-database}")
    public void handleTaskStatusMessage(UploadMessage message) {
        try {
            uploadRecordService.updateStatus(message.getTaskId(), "COMPLETED");
            log.info("记录已入库: {}", message.getFileName());

        } catch (Exception e) {
            try {
                uploadRecordService.updateStatus(message.getTaskId(), "FAILED");
            } catch (Exception ex) {
                log.error("设置失败状态时出错", ex);
            }
            log.error("入库处理异常", e);
        }
    }

    private String runPythonScript(String interpreter, String scriptPath, String... args) throws IOException, InterruptedException {
        List<String> command = new ArrayList<>();
        command.add(interpreter);
        command.add(scriptPath);
        command.addAll(Arrays.asList(args));
        ProcessBuilder pb = new ProcessBuilder(command);
        pb.redirectErrorStream(true);
        Process process = pb.start();
        String resultPath;
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
            resultPath = reader.readLine();
        }
        int exitCode = process.waitFor();
        if (exitCode != 0 || resultPath == null || resultPath.isBlank()) {
            return null;
        }
        return resultPath.trim();
    }
}
