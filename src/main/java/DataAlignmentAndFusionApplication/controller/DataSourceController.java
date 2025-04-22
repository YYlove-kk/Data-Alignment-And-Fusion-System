package DataAlignmentAndFusionApplication.controller;

import DataAlignmentAndFusionApplication.model.dto.DataSourceDTO;
import DataAlignmentAndFusionApplication.model.vo.DataSourceVO;
import DataAlignmentAndFusionApplication.service.DataSourceService;
import DataAlignmentAndFusionApplication.util.Result;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/data-sources")
public class DataSourceController {
    private final DataSourceService dataSourceService;

    public DataSourceController(DataSourceService dataSourceService) {
        this.dataSourceService = dataSourceService;
    }

    @PostMapping
    public Result<DataSourceVO> create(@RequestBody DataSourceDTO dto) {
//        return dataSourceService.createDataSource(dto);
        return null;
    }

    @DeleteMapping("/{id}")
    public Result<Void> delete(@PathVariable Long id) {
//        return dataSourceService.deleteDataSource(id);
        return null;
    }

    @GetMapping
    public Result<Page<DataSourceVO>> list(
            @RequestParam(defaultValue = "1") Integer page,
            @RequestParam(defaultValue = "10") Integer size,
            @RequestParam(required = false) String keyword) {
//        return dataSourceService.listDataSources(page, size, keyword);
        return null;
    }
}

