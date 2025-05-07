package DataAlignmentAndFusionApplication.service.impl;

import DataAlignmentAndFusionApplication.mapper.TrainRecordMapper;
import DataAlignmentAndFusionApplication.util.Result;
import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import DataAlignmentAndFusionApplication.model.entity.TrainRecord;
import DataAlignmentAndFusionApplication.service.TrainRecordService;
import com.opencsv.CSVReader;
import com.opencsv.exceptions.CsvException;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.io.*;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;

/**
* @author 29857
* @description 针对表【train_record】的数据库操作Service实现
* @createDate 2025-05-07 23:16:58
*/
@Service
public class TrainRecordServiceImpl extends ServiceImpl<TrainRecordMapper, TrainRecord>
    implements TrainRecordService {
    @Autowired
    private TrainRecordMapper trainRecordMapper;

    @Override
    public Result<String> startTraining() {
        String csvDir = "data/train";
        // 启动异步任务（建议使用线程池或 Spring @Async）
        new Thread(() -> runTraining(csvDir)).start();
        return Result.success("训练任务已启动");
    }

    private void runTraining(String csvDir) {
        try {
            ProcessBuilder pb = new ProcessBuilder("python", "DAFSPython/train/train_han.py");
            pb.redirectErrorStream(true);
            Process process = pb.start();

            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            StringBuilder output = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                output.append(line).append("\n");
            }

            int code = process.waitFor();

            File dir = new File(csvDir);
            if (!dir.exists() || !dir.isDirectory()) {
                throw new IllegalArgumentException("CSV目录不存在或不是目录: " + csvDir);
            }

            File[] csvFiles = dir.listFiles((d, name) -> name.endsWith(".csv"));
            if (csvFiles == null || csvFiles.length == 0) {
                throw new IllegalArgumentException("目录下未找到CSV文件");
            }

            for (File csvFile : csvFiles) {
                List<TrainRecord> records = parseCsvFile(csvFile);
                for (TrainRecord record : records) {
                    // 去重逻辑：判断是否已有相同 epoch + resultPath 的记录
                    QueryWrapper<TrainRecord> query = new QueryWrapper<>();
                    query.eq("epoch", record.getEpoch()).eq("result_path", record.getResultPath());

                    if (trainRecordMapper.selectCount(query) == 0) {
                        trainRecordMapper.insert(record);
                    }
                }
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // 读取 CSV 文件并填充 TrainRecord
    private List<TrainRecord> parseCsvFile(File csvFile) throws IOException {
        List<TrainRecord> records = new ArrayList<>();
        try (CSVReader csvReader = new CSVReader(new FileReader(csvFile))) {
            List<String[]> rows = csvReader.readAll();

            for (int i = 1; i < rows.size(); i++) { // 从1开始跳过header
                String[] row = rows.get(i);
                TrainRecord record = new TrainRecord();

                record.setEpoch(Integer.parseInt(row[0]));
                record.setResultPath(row[1]);
                record.setHits1(row[2]);
                record.setHits5(row[3]);
                record.setHits10(row[4]);
                record.setTrainLoss(row[5]);
                record.setTestLoss(row[6]);

                DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
                record.setCreateTime(LocalDateTime.parse(row[7], formatter));

                record.setStatus("SUCCESS"); // 可选，给定默认状态
                record.setLog("来自CSV批量导入");

                records.add(record);
            }
        } catch (IOException | CsvException e) {
            throw new IOException("Error reading CSV file: " + e.getMessage());
        }
        return records;
    }

    @Override
    public List<TrainRecord> getAll() {
        return trainRecordMapper.selectList(null); // 查询所有记录
    }

}



