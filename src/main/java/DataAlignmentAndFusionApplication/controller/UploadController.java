package DataAlignmentAndFusionApplication.controller;

import DataAlignmentAndFusionApplication.model.DTO.FileUploadDTO;
import DataAlignmentAndFusionApplication.model.DTO.UploadResultDTO;
import DataAlignmentAndFusionApplication.model.vo.UploadRecordVO;
import DataAlignmentAndFusionApplication.service.UploadService;
import DataAlignmentAndFusionApplication.util.Result;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/upload")
public class UploadController {
    @Autowired
    private UploadService uploadService;

    // 文件上传接口
    @PostMapping("/file")
    public Result<UploadResultDTO> uploadFile(@ModelAttribute FileUploadDTO dto) {
        return uploadService.upload(dto);
    }

    // 获取已上传列表
    @GetMapping("/list")
    public Result<Page<UploadRecordVO>> getUploadList(
            @RequestParam(required = false) Integer page,
            @RequestParam(required = false) Integer size) {
        return null;
    }

    // 删除记录
    @DeleteMapping("/{id}")
    public Result<Void> deleteRecord(@PathVariable Long id) {
        return null;
    }
}
