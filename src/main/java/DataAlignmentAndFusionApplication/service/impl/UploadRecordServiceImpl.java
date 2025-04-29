package DataAlignmentAndFusionApplication.service.impl;

import DataAlignmentAndFusionApplication.config.AppConfig;
import DataAlignmentAndFusionApplication.model.dto.FileUploadDTO;
import DataAlignmentAndFusionApplication.model.dto.UploadMessage;
import DataAlignmentAndFusionApplication.model.entity.UploadRecord;
import DataAlignmentAndFusionApplication.model.vo.PageVO;
import DataAlignmentAndFusionApplication.util.Result;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.UpdateWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import DataAlignmentAndFusionApplication.service.UploadRecordService;
import DataAlignmentAndFusionApplication.mapper.UploadRecordMapper;
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.Date;
import java.util.List;
import java.util.UUID;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

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
    public Result<String> uploadFile(FileUploadDTO dto) {
        try {
            String taskId = UUID.randomUUID().toString();
            String patientId = dto.getPatientId();
            MultipartFile file = dto.getFile();
            String fileName = file.getOriginalFilename();
            String suffix = fileName.substring(fileName.lastIndexOf('.') + 1).toLowerCase();

            Path patientDir = rawDir.resolve(patientId);
            Path targetDir;

            if (suffix.equals("xlsx") || suffix.equals("xls") || suffix.equals("csv")) {
                targetDir = patientDir.resolve("text");
            } else if (suffix.equals("zip")) {
                targetDir = patientDir.resolve("image");
            } else {
                return Result.error(500, "文件类型不支持");
            }

            Files.createDirectories(targetDir);

            String rawPath;
            if (suffix.equals("zip")) {
                Path zipPath = targetDir.resolve(fileName);
                Files.copy(file.getInputStream(), zipPath, StandardCopyOption.REPLACE_EXISTING);
                unzip(zipPath.toFile(), targetDir.toFile());
                Files.deleteIfExists(zipPath);
                rawPath = targetDir.toString();
            } else {
                Path targetPath = targetDir.resolve(fileName);
                Files.copy(file.getInputStream(), targetPath, StandardCopyOption.REPLACE_EXISTING);
                rawPath = targetPath.toString();
            }

            UploadMessage message = UploadMessage.builder()
                    .rawPath(rawPath)
                    .schemaRegistryPath(schemaRegistry.toString())
                    .reportDir(reportDir.toString())
                    .cleanDir(cleanDir.toString())
                    .outputDir(outputDir.toString())
                    .fileName(fileName)
                    .institution(dto.getInstitution())
                    .modalityType(dto.getModalityType())
                    .taskId(taskId)
                    .patientId(patientId)
                    .status("WAITING")
                    .build();

            // 创建任务记录，但不发送
            createWaitingRecord(message);

            return Result.success(taskId); // 前端拿到 taskId

        } catch (IOException e) {
            e.printStackTrace();
            return Result.error(500, "上传失败：" + e.getMessage());
        }
    }

    @Override
    public Result<String> embedFile() {
        // 1. 查询所有状态为 WAITING 的记录
        List<UploadRecord> waitingRecords = uploadRecordMapper.selectList(
                new LambdaQueryWrapper<UploadRecord>()
                        .eq(UploadRecord::getStatus, "WAITING")
        );

        if (waitingRecords.isEmpty()) {
            return Result.success("没有需要处理的任务");
        }

        for (UploadRecord record : waitingRecords) {
            UploadMessage message = UploadMessage.builder()
                    .taskId(record.getTaskId())
                    .rawPath(record.getRawPath())
                    .schemaRegistryPath(schemaRegistry.toString())
                    .reportDir(reportDir.toString())
                    .cleanDir(cleanDir.toString())
                    .outputDir(outputDir.toString())
                    .fileName(record.getFileName())
                    .institution(record.getInstitution())
                    .modalityType(record.getModalityType())
                    .patientId(record.getPatientId())
                    .status("PROCESSING")
                    .build();

            // 2. 更新状态
            updateStatus(record.getTaskId(), "PROCESSING");

            // 3. 发送消息
            rabbitTemplate.convertAndSend(uploadToCleaningQueue, message);
        }

        return Result.success("成功处理任务数量：" + waitingRecords.size());
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
        record.setPatientId(message.getPatientId());
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

    @Override
    public void updatePaths(String taskId, String cleanPath, String outputPath) {
        if (taskId == null || cleanPath == null || outputPath == null) {
            throw new IllegalArgumentException("taskId, cleanPath,outputPath有空值");
        }

        // 从数据库中获取记录
        UploadRecord record = uploadRecordMapper.selectById(taskId);
        if (record == null) {
            throw new RuntimeException("找不到" + taskId);
        }

        // 更新路径
        record.setCleanPath(cleanPath);
        record.setOutputPath(outputPath);

        // 更新记录
        int updatedRows = uploadRecordMapper.updateById(record);
        if (updatedRows == 0) {
            throw new RuntimeException("更新失败" + taskId);
        }
    }

    public void unzip(File zipFile, File targetDir) {
        try (ZipInputStream zipInputStream = new ZipInputStream(new FileInputStream(zipFile))) {
            ZipEntry entry;
            while ((entry = zipInputStream.getNextEntry()) != null) {
                File entryFile = new File(targetDir, entry.getName());
                if (entry.isDirectory()) {
                    entryFile.mkdirs();
                } else {
                    try (FileOutputStream fos = new FileOutputStream(entryFile)) {
                        byte[] buffer = new byte[1024];
                        int len;
                        while ((len = zipInputStream.read(buffer)) > 0) {
                            fos.write(buffer, 0, len);
                        }
                    }
                }
                zipInputStream.closeEntry();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
