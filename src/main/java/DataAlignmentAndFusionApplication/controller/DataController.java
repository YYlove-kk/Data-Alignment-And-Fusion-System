package DataAlignmentAndFusionApplication.controller;
import DataAlignmentAndFusionApplication.common.UploadResp;
import DataAlignmentAndFusionApplication.model.dto.DataSourceDTO;
import DataAlignmentAndFusionApplication.model.dto.FileUploadDTO;
import DataAlignmentAndFusionApplication.model.dto.UploadResultDTO;
import DataAlignmentAndFusionApplication.model.vo.DataSourceVO;
import DataAlignmentAndFusionApplication.model.vo.UploadRecordVO;
import DataAlignmentAndFusionApplication.service.DataSourceService;
import DataAlignmentAndFusionApplication.service.UploadRecordService;
import DataAlignmentAndFusionApplication.util.Result;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api")
public class DataController {
    private DataSourceService dataSourceService;
    private UploadRecordService uploadService;



    // 上传文件并创建数据源
    @PostMapping("/upload-data")
    public Result<DataSourceVO> uploadFileAndCreateDataSource(@ModelAttribute FileUploadDTO dto) {
        // 调用上传服务
        UploadResp uploadResult = uploadService.uploadFileAndProcess(dto);

        // 根据上传结果创建数据源
        DataSourceDTO dataSourceDTO = new DataSourceDTO();
        // 假设上传结果包含文件路径，将其设置到数据源 DTO 中
        dataSourceDTO.setFilePath(uploadResult.getRawPath());

        return dataSourceService.createDataSource(dataSourceDTO);
    }
    // 删除数据源
    @DeleteMapping("/data/{id}")
    public Result<Void> delete(@PathVariable Long id) {
         return dataSourceService.deleteDataSource(id);

    }
    // 列出数据源
    @GetMapping("/data")
    public Result<Page<DataSourceVO>> list(
            @RequestParam(defaultValue = "1") Integer page,
            @RequestParam(defaultValue = "10") Integer size,
            @RequestParam(required = false) String keyword) {
         return dataSourceService.listDataSources(page, size, keyword);

    }
    // 获取已上传列表
    @GetMapping("/upload/list")
    public Result<Page<UploadRecordVO>> getUploadList(
            @RequestParam(required = false) Integer page,
            @RequestParam(required = false) Integer size) {
        return null;
    }
    // 删除上传记录
    @DeleteMapping("/upload/{id}")
    public Result<Void> deleteUploadRecord(@PathVariable Long id) {
        return null;
    }
}
