package DataAlignmentAndFusionApplication.service.impl;

import DataAlignmentAndFusionApplication.common.UploadResp;
import DataAlignmentAndFusionApplication.config.AppConfig;
import DataAlignmentAndFusionApplication.model.dto.FileUploadDTO;
import DataAlignmentAndFusionApplication.model.dto.UploadMessage;
import DataAlignmentAndFusionApplication.model.entity.UploadRecord;
import DataAlignmentAndFusionApplication.util.Result;
import com.baomidou.mybatisplus.core.conditions.update.UpdateWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import DataAlignmentAndFusionApplication.service.UploadRecordService;
import DataAlignmentAndFusionApplication.mapper.module.UploadRecordMapper;
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.Date;
import java.util.Objects;
import java.util.UUID;

/**
 * @author 29857
 * @description 针对表【upload_record】的数据库操作Service实现
 * @createDate 2025-04-17 20:05:12
 */
@Service
public class UploadRecordServiceImpl extends ServiceImpl<UploadRecordMapper, UploadRecord> implements UploadRecordService {


    @Autowired
    private UploadRecordMapper uploadRecordMapper;

    @Autowired
    private RabbitTemplate rabbitTemplate;

    @Value("${mq.upload-to-cleaning}")
    private String uploadToCleaningQueue;

    private final AppConfig appConfig;

    private final Path rawDir;
    private final Path cleanDir;
    private final Path reportDir;
    private final Path outputDir;

    @Autowired
    public UploadRecordServiceImpl(AppConfig appConfig, UploadRecordMapper uploadRecordMapper, RabbitTemplate rabbitTemplate) {
        this.appConfig = appConfig;
        this.rawDir = Paths.get(appConfig.getUploadRawDir());
        this.cleanDir = Paths.get(appConfig.getUploadCleanDir());
        this.reportDir = Paths.get(appConfig.getUploadReportDir());
        this.outputDir = Paths.get(appConfig.getUploadOutputDir());
    }

    private UploadResp processDicomFile(UploadMessage message) {
        // 处理 DICOM 文件的逻辑
        return null;
    }

    @Override
    public Result<String> uploadFileAndProcess(FileUploadDTO dto) {

        try {
            // 生成 taskId
            String taskId = UUID.randomUUID().toString();
            // 1. 保存原文件
            MultipartFile file = dto.getFile();
            String fileName = file.getOriginalFilename();
            String rawPath = String.valueOf(Files.copy(file.getInputStream(),
                    rawDir.resolve(Objects.requireNonNull(fileName)),
                    StandardCopyOption.REPLACE_EXISTING));
            // 2. 调用 Python 清洗
            UploadMessage message = new UploadMessage();
            message.setRawPath(rawPath);
            message.setSchemaRegistryPath(dto.getSchemaRegistryPath());
            message.setReportDir(reportDir.toString());
            message.setOutputDir(outputDir.toString());
            message.setFileName(fileName);
            message.setTaskId(taskId);
            message.setStatus("WAITING");

            // 验证文件类型
            if (fileName.endsWith(".xlsx") || fileName.endsWith(".xls") || fileName.endsWith(".csv")) {
                // 处理 Excel 文件
                UploadRecord record = uploadRecordMapper.selectById(message.getTaskId());
                if (record != null) {
                    record.setStatus("PROCESSING");  // 设置为处理中
                    uploadRecordMapper.updateById(record);
                }
                rabbitTemplate.convertAndSend(uploadToCleaningQueue, message);
                return Result.success(taskId);

            } else if (fileName.endsWith(".dcm")) {
                // 处理 DICOM 文件
                return Result.success(taskId);
            } else {
                // 文件类型不支持
                return Result.error(500, "文件类型不支持");
            }
        } catch (IOException e) {
            e.printStackTrace();
            return Result.error(500, "上传失败：" + e.getMessage());
        }
    }

    @Override
    public Result<Page<UploadRecord>> getUploadList(Integer page, Integer size) {

//        Page<UploadRecordVO> uploadRecordVOPage = new PageImpl<>(Collections.emptyList(), PageRequest.of(page - 1, size), 0);
//        return Result.success(uploadRecordVOPage);
        return null;
    }

    @Override
    public Result<Void> deleteRecord(Long id) {
        // 实现删除记录逻辑
        // 这里只是示例，你需要根据实际情况实现具体逻辑
        return Result.success(null);
    }

    @Override
    public void createWaitingRecord(UploadMessage message) {
        UploadRecord record = new UploadRecord();
        record.setTaskId(message.getTaskId());
        record.setRawPath(message.getRawPath());
        record.setSchemaRegistryPath(message.getSchemaRegistryPath());
        record.setReportDir(message.getReportDir());
        record.setOutputDir(message.getOutputDir());
        record.setProcessTime(new Date());
        record.setStatus("WAITING");

        uploadRecordMapper.insert(record);
    }

    @Override
    public void updateStatus(String taskId, String status) {
        UploadRecord record = new UploadRecord();
        record.setStatus(status);

        // 更新任务状态
        UpdateWrapper<UploadRecord> updateWrapper = new UpdateWrapper<>();
        updateWrapper.eq("task_id", taskId);
        uploadRecordMapper.update(record, updateWrapper);

    }
}
