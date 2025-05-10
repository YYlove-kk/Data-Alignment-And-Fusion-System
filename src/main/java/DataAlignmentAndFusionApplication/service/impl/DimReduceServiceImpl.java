package DataAlignmentAndFusionApplication.service.impl;

import DataAlignmentAndFusionApplication.config.AppConfig;
import DataAlignmentAndFusionApplication.mapper.EmbedRecordMapper;
import DataAlignmentAndFusionApplication.mapper.ReduceRecordMapper;
import DataAlignmentAndFusionApplication.mapper.UploadRecordMapper;
import DataAlignmentAndFusionApplication.model.entity.EmbedRecord;
import DataAlignmentAndFusionApplication.model.entity.ReduceRecord;
import DataAlignmentAndFusionApplication.model.entity.UploadRecord;
import DataAlignmentAndFusionApplication.service.DimReduceService;
import DataAlignmentAndFusionApplication.service.UserService;
import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

@Service
public class DimReduceServiceImpl extends ServiceImpl<ReduceRecordMapper, ReduceRecord> implements DimReduceService {

    @Autowired
    private AppConfig appConfig;

    @Autowired
    private ReduceRecordMapper reduceRecordMapper;

    @Autowired
    private EmbedRecordMapper embedRecordMapper;

    @Override
    public void reduce() {
        try {
            // 构建Python脚本命令，传递参数
            String interpreter = appConfig.getInterpreterPath();
            String scriptPath = appConfig.getKpcaReducePath();
            String inDir = appConfig.getAlignOutputPath();
            String outDir = appConfig.getAlignOutputPath() + "reduce/";

            File dir = new File(inDir);
            List<String> fileNames = new ArrayList<>();

            if (dir.exists() && dir.isDirectory()) {
                File[] files = dir.listFiles((d, name) -> name.toLowerCase().endsWith(".npy"));
                if (files != null) {
                    for (File file : files) {
                        fileNames.add(file.getName());
                    }
                }
            }

            for (String fileName : fileNames) {
                List<EmbedRecord> records = embedRecordMapper.selectList(
                        new QueryWrapper<EmbedRecord>().eq("npy_name", fileName)
                );
                if (!records.isEmpty()) {
                    String patientId = fileName.split("_")[0];
                    String sourceId = records.get(0).getSourceId();
                    ReduceRecord reduceRecord = new ReduceRecord();
                    reduceRecord.setFilename(fileName);
                    reduceRecord.setSourceId(sourceId);
                    reduceRecord.setStatus("COMPLETED");
                    reduceRecord.setPatientId(patientId);
                    reduceRecordMapper.insert(reduceRecord);
                }
            }

            String command = String.format("%s %s --input_dir %s --output_dir %s", interpreter, scriptPath, inDir, outDir);

            // 执行 Python 脚本
            Process process = Runtime.getRuntime().exec(command);

            // 等待脚本执行完成
            int exitCode = process.waitFor();
            if (exitCode == 0) {
                System.out.println("Dimension reduction completed successfully.");
            } else {
                throw new RuntimeException("Dimension reduction failed with exit code: " + exitCode);
            }
        } catch (IOException | InterruptedException e) {
            // 异常处理
            throw new RuntimeException("Error occurred while executing the Python script: " + e.getMessage(), e);
        }
    }

    @Override
    public List<ReduceRecord> getRecords() {
        return reduceRecordMapper.selectList(null);
    }

    @Override
    public List<String> getSourceIds() {
        return getRecords().stream().map(ReduceRecord::getSourceId).filter(Objects::nonNull).distinct().toList();
    }


}

