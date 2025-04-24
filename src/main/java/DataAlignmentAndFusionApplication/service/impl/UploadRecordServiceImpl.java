package DataAlignmentAndFusionApplication.service.impl;

import DataAlignmentAndFusionApplication.common.UploadResp;
import DataAlignmentAndFusionApplication.model.dto.FileUploadDTO;
import DataAlignmentAndFusionApplication.model.vo.UploadRecordVO;
import DataAlignmentAndFusionApplication.util.Result;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import DataAlignmentAndFusionApplication.model.entity.UploadRecord;
import DataAlignmentAndFusionApplication.service.UploadRecordService;
import DataAlignmentAndFusionApplication.mapper.module.UploadRecordMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;

/**
* @author 29857
* @description 针对表【upload_record】的数据库操作Service实现
* @createDate 2025-04-17 20:05:12
*/
@Service
public class UploadRecordServiceImpl extends ServiceImpl<UploadRecordMapper, UploadRecord>
    implements UploadRecordService{

//    @Value("${schema.registry}")
    private String schemaRegistry;


    private final Path rawDir = Paths.get("/data/raw");
    private final Path cleanDir = Paths.get("/data/clean");

    @Override
    public UploadResp uploadFileAndProcess(FileUploadDTO dto) {
        MultipartFile file = dto.getFile();
        try {
            // 1. 保存原文件
            String rawPath = String.valueOf(Files.copy(file.getInputStream(),
                    rawDir.resolve(file.getOriginalFilename()),
                    StandardCopyOption.REPLACE_EXISTING));
            // 2. 调用 Python 清洗
            ProcessBuilder pb = new ProcessBuilder(
                    "python", "data_ingest.py",
                    "--file", rawPath,
                    "--schema", schemaRegistry
            );
            Process p = pb.start();
            String cleanPath = new BufferedReader(
                    new InputStreamReader(p.getInputStream())
            ).readLine();

            // 3. 入库元数据（略）
            // 4. 返回给前端
            return new UploadResp(rawPath, cleanPath, "SUCCESS");
        } catch (IOException e) {
            e.printStackTrace();
            return new UploadResp(null, null, "ERROR");
        }
    }
    @Override
    public Result<Page<UploadRecordVO>> getUploadList(Integer page, Integer size) {

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
}




