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
import java.io.InputStreamReader;

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
            String ingestPath = appConfig.getDataIngestPath();
            String scriptPath;
            if (fileName.endsWith(".dcm")) {
                scriptPath = appConfig.getImageEmbedPath(); // 针对 DICOM 图像

                ProcessBuilder pb = new ProcessBuilder(
                        interpreter, scriptPath,
                        "--file_path", message.getRawPath(),
                        "--output_dir", message.getOutputDir()
                );
                pb.redirectErrorStream(true);
                Process p = pb.start();
                String outputPath = null;
                try (BufferedReader reader = new BufferedReader(new InputStreamReader(p.getInputStream()))) {
                    outputPath = reader.readLine();
                }

                int exitCode = p.waitFor();
                if (exitCode != 0 || outputPath == null || outputPath.isBlank()) {
                    log.error("Python嵌入失败: 文件={}, 返回路径={}", message.getRawPath(), outputPath);
                    return;
                }
                // 成功处理，设置嵌入后的路径
                message.setCleanPath(outputPath.trim());  // trim一下保险
                rabbitTemplate.convertAndSend(cleaningToDatabaseQueue, message);
            } else {
                // 先清洗
                ProcessBuilder cleanPb = new ProcessBuilder(
                        interpreter, ingestPath,
                        "--file_path", message.getRawPath(),
                        "--clean_dir", message.getOutputDir(),
                        "--registry_path", message.getSchemaRegistryPath(),
                        "--report_dir", message.getReportDir()
                );
                cleanPb.redirectErrorStream(true);
                Process cleanP = cleanPb.start();
                String cleanPath = null;
                try (BufferedReader reader = new BufferedReader(new InputStreamReader(cleanP.getInputStream()))) {
                    cleanPath = reader.readLine();
                }
                int exitCode = cleanP.waitFor();
                if (exitCode != 0 || cleanPath == null || cleanPath.isBlank()) {
                    log.error("Python清洗失败: 文件={}, 返回路径={}", message.getRawPath(), cleanPath);
                    return;
                }
                // 成功处理，设置清洗后的路径
                message.setCleanPath(cleanPath.trim());  // trim一下保险
                rabbitTemplate.convertAndSend(cleaningToDatabaseQueue, message);

                scriptPath = appConfig.getTextTimeEmbedPath(); // 数据嵌入

                ProcessBuilder embedPb = new ProcessBuilder(
                        interpreter, scriptPath,
                        "--file_path", message.getCleanPath(),
                        "--clean_dir", message.getOutputDir(),
                        "--registry_path", message.getSchemaRegistryPath(),
                        "--report_dir", message.getReportDir()
                );
                embedPb.redirectErrorStream(true);
                Process embedP = embedPb.start();
                String outputPath = null;
                try (BufferedReader reader = new BufferedReader(new InputStreamReader(embedP.getInputStream()))) {
                    outputPath = reader.readLine();
                }

                int exitCode2 = embedP.waitFor();
                if (exitCode2 != 0 || outputPath == null || outputPath.isBlank()) {
                    log.error("Python嵌入失败: 文件={}, 返回路径={}", message.getRawPath(), outputPath);
                }
            }


        } catch (Exception e) {
            log.error("清洗处理异常: 文件={}", message.getRawPath(), e);
        }
    }

    @RabbitListener(queues = "${mq.cleaning-to-database}")
    public void handleTaskStatusMessage(UploadMessage message) {
        try {
            uploadRecordService.updateStatus(message.getTaskId(), "COMPLETED");
            log.info("清洗记录已入库: {}", message.getFileName());

        } catch (Exception e) {
            try {
                uploadRecordService.updateStatus(message.getTaskId(), "FAILED");
            } catch (Exception ex) {
                log.error("设置失败状态时出错", ex);
            }
            log.error("入库处理异常", e);
        }
    }
}
