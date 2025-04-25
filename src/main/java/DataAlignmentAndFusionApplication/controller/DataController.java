package DataAlignmentAndFusionApplication.controller;
import DataAlignmentAndFusionApplication.model.dto.FileUploadDTO;
import DataAlignmentAndFusionApplication.model.entity.UploadRecord;
import DataAlignmentAndFusionApplication.model.vo.DataSourceVO;

import DataAlignmentAndFusionApplication.service.UploadRecordService;
import DataAlignmentAndFusionApplication.util.Result;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/upload")
public class DataController {
    private UploadRecordService uploadService;

    // 上传文件
    @PostMapping("/data")
    public Result<String> uploadFile(@ModelAttribute FileUploadDTO dto) {
        // 调用上传服务
        return uploadService.uploadFileAndProcess(dto);
    }

    // 获取已上传列表
    @GetMapping("/list")
    public Result<Page<UploadRecord>> getUploadList(
            @RequestParam(required = false) Integer page,
            @RequestParam(required = false) Integer size) {
        return uploadService.getUploadList(page, size);
    }
    // 删除上传记录
    @DeleteMapping("/{id}")
    public Result<Void> deleteUploadRecord(@PathVariable Long id) {
        return uploadService.deleteRecord(id);
    }
}
