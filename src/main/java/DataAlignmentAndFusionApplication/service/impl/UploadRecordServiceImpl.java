package DataAlignmentAndFusionApplication.service.impl;

import DataAlignmentAndFusionApplication.config.AppConfig;
import DataAlignmentAndFusionApplication.model.dto.FileUploadDTO;
import DataAlignmentAndFusionApplication.model.dto.UploadMessage;
import DataAlignmentAndFusionApplication.model.entity.UploadRecord;
import DataAlignmentAndFusionApplication.model.vo.PageVO;
import DataAlignmentAndFusionApplication.util.Result;
import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
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
    private final Path reportDir;
    private final Path cleanDir;
    private final Path outputDir;
    private final Path schemaRegistry;

    @Autowired
    public UploadRecordServiceImpl(AppConfig appConfig, UploadRecordMapper uploadRecordMapper, RabbitTemplate rabbitTemplate) {
        this.appConfig = appConfig;
        this.rawDir = Paths.get(appConfig.getUploadRawDir());
        this.reportDir = Paths.get(appConfig.getUploadReportDir());
        this.cleanDir = Paths.get(appConfig.getUploadCleanDir());
        this.outputDir = Paths.get(appConfig.getUploadOutputDir());
        this.schemaRegistry = Paths.get(appConfig.getSchemaRegistryPath());

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
            message.setSchemaRegistryPath(schemaRegistry.toString());
            message.setReportDir(reportDir.toString());
            message.setCleanDir(cleanDir.toString());
            message.setOutputDir(outputDir.toString());
            message.setFileName(fileName);
            message.setInstitution(dto.getInstitution());
            message.setModalityType(dto.getModalityType());
            message.setTaskId(taskId);
            message.setStatus("WAITING");

            // 验证文件类型
            if (fileName.endsWith(".xlsx") || fileName.endsWith(".xls") || fileName.endsWith(".csv") || fileName.endsWith(".dcm")) {
                // 处理 Excel 文件
                // 1. 创建新的任务记录，初始状态是 WAITING
                createWaitingRecord(message);
                // 2. 立即将状态更新为 PROCESSING
                updateStatus(message.getTaskId(), "PROCESSING");
                rabbitTemplate.convertAndSend(uploadToCleaningQueue, message);
                //任务创建成功
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
    public Result<UploadRecord> getTaskStatus(String taskId) {
        UploadRecord record = uploadRecordMapper.selectOne(
                new QueryWrapper<UploadRecord>().eq("task_id", taskId).eq("status", "COMPLETED")
        );
        if (record == null) {
            return Result.error(404, "任务不存在");
        }
        return Result.success(record);
    }

    @Override
    public Result<PageVO<UploadRecord>> getUploadList(Integer page, Integer size) {
        // 1. 分页查询 UploadRecord
        Page<UploadRecord> uploadRecordPage = uploadRecordMapper.selectPage(
                new Page<>(page, size),
                null  // 可以根据需要添加查询条件，比如按用户、按时间等
        );

        // 2. 封装成 PageVO
        PageVO<UploadRecord> pageVO = new PageVO<>();
        pageVO.setList(uploadRecordPage.getRecords());
        pageVO.setTotal(uploadRecordPage.getTotal());
        pageVO.setPageSize((int) uploadRecordPage.getSize());
        pageVO.setCurrentPage((int) uploadRecordPage.getCurrent());

        // 3. 返回结果
        return Result.success(pageVO);
    }

    @Override
    public Result<Void> deleteRecord(Long id) {
        if (id == null) {
            return Result.error(500, "ID不能为空");
        }

        if (uploadRecordMapper.selectById(id) == null) {
            return Result.error(404, "记录不存在");
        }

        int deleted = uploadRecordMapper.deleteById(id);

        if (deleted > 0) {
            return Result.success(null);  // 删除成功
        } else {
            return Result.error(500, "记录不存在或已被删除");  // 删除失败，比如ID查不到
        }
    }

    @Override
    public void createWaitingRecord(UploadMessage message) {
        UploadRecord record = new UploadRecord();
        record.setTaskId(message.getTaskId());
        record.setRawPath(message.getRawPath());
        record.setSchemaRegistryPath(message.getSchemaRegistryPath());
        record.setReportDir(message.getReportDir());
        record.setCleanDir(message.getCleanDir());
        record.setOutputDir(message.getOutputDir());
        record.setInstitution(message.getInstitution());
        record.setModalityType(message.getModalityType());
        record.setProcessTime(new Date());
        record.setStatus("WAITING");

        uploadRecordMapper.insert(record);
    }

    @Override
    public void updateStatus(String taskId, String status) {
        UpdateWrapper<UploadRecord> updateWrapper = new UpdateWrapper<>();
        updateWrapper.eq("task_id", taskId)
                .set("status", status); // 直接只更新 status 字段
        uploadRecordMapper.update(null, updateWrapper);
    }
}
