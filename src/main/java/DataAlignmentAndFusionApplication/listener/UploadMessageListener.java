package DataAlignmentAndFusionApplication.listener;

import DataAlignmentAndFusionApplication.config.AppConfig;
import DataAlignmentAndFusionApplication.mapper.EmbedRecordMapper;
import DataAlignmentAndFusionApplication.mapper.UploadRecordMapper;
import DataAlignmentAndFusionApplication.model.dto.UploadMessage;
import DataAlignmentAndFusionApplication.model.entity.EmbedRecord;
import DataAlignmentAndFusionApplication.service.EmbedRecordService;
import DataAlignmentAndFusionApplication.service.UploadRecordService;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
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
import java.util.stream.Collectors;

@Slf4j
@Component
public class UploadMessageListener {


    @Autowired
    private RabbitTemplate rabbitTemplate;

    @Autowired
    private UploadRecordService uploadRecordService;

    @Autowired
    private EmbedRecordService embedRecordService;


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
            String modalityType = message.getModalityType();
            String scriptPath;
            List<String> paths = new ArrayList<>();

            if (modalityType.equals("IMAGE")) {
                scriptPath = appConfig.getImageEmbedPath(); // 针对 DICOM 图像
                String outputPath = runPythonScript(interpreter, scriptPath,
                        "--source_folder", "../" + message.getRawDir(),
                        "--output_dir", "../data/align/output/image");

                if (outputPath != null) {
                    ObjectMapper mapper = new ObjectMapper();
                    JsonNode root = mapper.readTree(outputPath);

                    if (root.has("paths")) {
                        for (JsonNode node : root.get("paths")) {
                            paths.add(node.asText());
                        }
                    }
                }
                // 成功处理，设置嵌入后的路径
                message.setCleanPath(message.getRawDir());
            } else {
                // 先清洗
                String ingestPath = appConfig.getDataIngestPath();
                String cleanPath = runPythonScript(interpreter, ingestPath,
                        "--file_path", "../" + message.getRawDir() + fileName,
                        "--clean_dir", "../" + message.getCleanDir(),
                        "--registry_path", "../" + message.getSchemaRegistryPath(),
                        "--report_dir", "../" + message.getReportDir());

                if (cleanPath == null) {
                    log.error("Python清洗失败: 文件={}", message.getRawDir());
                    return;
                }
                // 清洗后的路径
                message.setCleanPath(cleanPath.trim());  // trim一下保险
                // 数据嵌入
                scriptPath = appConfig.getTextTimeEmbedPath();

                String outputPath = runPythonScript(interpreter, scriptPath,
                        "--file_path", "../" + message.getCleanPath(),
                        "--output_dir", "../data/align/output/text");

                if (outputPath != null) {
                    ObjectMapper mapper = new ObjectMapper();
                    JsonNode root = mapper.readTree(outputPath);

                    if (root.has("paths")) {
                        for (JsonNode node : root.get("paths")) {
                            paths.add(node.asText());
                        }
                    }
                }

            }

            List<EmbedRecord> records = paths.stream().map(path -> {
                EmbedRecord record = new EmbedRecord();
                record.setNpyName(path);
                record.setSourceId(message.getSourceId());
                return record;
            }).collect(Collectors.toList());

            if (!records.isEmpty()) {
                embedRecordService.saveBatch(records);
            }

            message.setSingleEmbedNpy(paths);
            uploadRecordService.updatePaths(message.getTaskId(), message.getCleanPath(), message.getSingleEmbedNpy());
            rabbitTemplate.convertAndSend(cleaningToDatabaseQueue, message);

        } catch (Exception e) {
            log.error("清洗处理异常: 文件={}", message.getRawDir(), e);
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
        pb.redirectErrorStream(true); // 合并 stderr + stdout
        Process process = pb.start();

        StringBuilder jsonBuilder = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
            String line;
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                // 简单判断是否为 JSON 行
                if (line.startsWith("{") && line.endsWith("}")) {
                    jsonBuilder.append(line);
                    break;  // 如果只输出一次 JSON，立即跳出
                }
            }
        }

        int exitCode = process.waitFor();
        String jsonOutput = jsonBuilder.toString().trim();
        if (exitCode != 0 || jsonOutput.isBlank()) {
            return null;
        }
        return jsonOutput;
    }

}
