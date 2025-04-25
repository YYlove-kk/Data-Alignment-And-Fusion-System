package DataAlignmentAndFusionApplication.listener;

import DataAlignmentAndFusionApplication.mapper.module.UploadRecordMapper;
import DataAlignmentAndFusionApplication.model.dto.UploadMessage;
import DataAlignmentAndFusionApplication.model.entity.UploadRecord;
import lombok.extern.slf4j.Slf4j;
import org.springframework.amqp.rabbit.annotation.RabbitListener;
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;


import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.time.LocalDateTime;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;

@Slf4j
@Component
public class UploadMessageListener {

    @Value("${algorithm.data-ingest}")
    private String dataIngest;

    @Autowired
    private RabbitTemplate rabbitTemplate;

    @Autowired
    private UploadRecordMapper uploadRecordMapper;

    @Value("${mq.cleaning-to-database}")
    private String cleaningToDatabaseQueue;

    @RabbitListener(queues = "${mq.upload-to-cleaning}")
    public void handleUploadMessage(UploadMessage message) {
        try {
            // 调用 Python
            ProcessBuilder pb = new ProcessBuilder(
                    "python", dataIngest,
                    "--file_path", message.getRawPath(),
                    "--registry_path", message.getSchemaRegistryPath(),
                    "--report_dir", message.getReportDir(),
                    "--output_dir", message.getOutputDir()
            );
            pb.redirectErrorStream(true);
            Process p = pb.start();
            String cleanPath = new BufferedReader(new InputStreamReader(p.getInputStream())).readLine();
            int exitCode = p.waitFor();
            if (exitCode != 0) {
                log.error("Python清洗失败: {}", message.getRawPath());
                return;
            }

            // 设置清洗后的路径并发送到下一队列
            message.setCleanPath(cleanPath);
            rabbitTemplate.convertAndSend(cleaningToDatabaseQueue, message);

        } catch (Exception e) {
            log.error("清洗处理异常", e);
        }
    }

    @RabbitListener(queues = "${mq.cleaning-to-database}")
    public void handleCleanedMessage(UploadMessage message) {
        try {
            UploadRecord record = new UploadRecord();
            record.setTaskId(message.getTaskId());
            record.setRawPath(message.getRawPath());
            record.setCleanPath(message.getCleanPath());
            record.setSchemaRegistryPath(message.getSchemaRegistryPath());
            record.setReportDir(message.getReportDir());
            record.setOutputDir(message.getOutputDir());
            record.setProcessTime(new Date());
            record.setStatus("COMPLETED");

            uploadRecordMapper.updateById(record);

            log.info("清洗记录已入库: {}", message.getFileName());
        } catch (Exception e) {

            UploadRecord record = uploadRecordMapper.selectById(message.getTaskId());
            if (record != null) {
                record.setStatus("FAILED");  // 设置为失败
                uploadRecordMapper.updateById(record);
            }
            log.error("入库处理异常", e);
        }
    }
}
