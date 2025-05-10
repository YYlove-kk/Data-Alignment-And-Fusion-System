package DataAlignmentAndFusionApplication.controller;
import DataAlignmentAndFusionApplication.model.dto.UploadReq;
import DataAlignmentAndFusionApplication.model.entity.UploadRecord;
import DataAlignmentAndFusionApplication.model.vo.PageVO;
import DataAlignmentAndFusionApplication.service.UploadRecordService;
import DataAlignmentAndFusionApplication.util.Result;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/data")
public class UploadController {
    @Autowired
    private UploadRecordService uploadRecordService;

    // 上传文件
    @PostMapping("/upload")
    public Result<String> uploadFile(@RequestBody UploadReq req) {
        // 调用上传服务
        return uploadRecordService.uploadFile(req);
    }

    @PostMapping("/embed")
    public Result<String> embedFile() {
        // 嵌入服务
        return uploadRecordService.embedFile();
    }

    // 获取已上传列表
    @GetMapping("/list")
    public Result<PageVO<UploadRecord>> getUploadList(
            @RequestParam(required = false) Integer page,
            @RequestParam(required = false) Integer size) {
        return uploadRecordService.getUploadList(page, size);
    }
    // 删除上传记录
    @DeleteMapping("/delete/{id}")
    public Result<Void> deleteUploadRecord(@PathVariable Long id) {
        return uploadRecordService.deleteRecord(id);
    }
}
