package DataAlignmentAndFusionApplication.controller;
import DataAlignmentAndFusionApplication.model.dto.FileUploadDTO;
import DataAlignmentAndFusionApplication.model.entity.UploadRecord;
import DataAlignmentAndFusionApplication.model.vo.PageVO;
import DataAlignmentAndFusionApplication.service.UploadRecordService;
import DataAlignmentAndFusionApplication.util.Result;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/upload")
public class DataController {
    private UploadRecordService uploadRecordService;

    // 上传文件
    @PostMapping("/data")
    public Result<String> uploadFile(@RequestBody FileUploadDTO dto) {
        // 调用上传服务
        return uploadRecordService.uploadFileAndProcess(dto);
    }

    // 获取已上传列表
    @GetMapping("/list")
    public Result<PageVO<UploadRecord>> getUploadList(
            @RequestParam(required = false) Integer page,
            @RequestParam(required = false) Integer size) {
        return uploadRecordService.getUploadList(page, size);
    }
    // 删除上传记录
    @DeleteMapping("/{id}")
    public Result<Void> deleteUploadRecord(@PathVariable Long id) {
        return uploadRecordService.deleteRecord(id);
    }
}
