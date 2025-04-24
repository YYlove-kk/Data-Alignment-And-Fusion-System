package DataAlignmentAndFusionApplication.service;

import DataAlignmentAndFusionApplication.model.dto.DataSourceDTO;
import DataAlignmentAndFusionApplication.model.entity.DataSource;
import DataAlignmentAndFusionApplication.model.vo.DataSourceVO;
import DataAlignmentAndFusionApplication.util.Result;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.IService;

/**
* @author 29857
* @description 针对表【data_source(数据源管理表)】的数据库操作Service
* @createDate 2025-04-17 20:02:26
*/
public interface DataSourceService extends IService<DataSource> {

    // 创建数据源
    Result<DataSourceVO> createDataSource(DataSourceDTO dto);

    // 删除数据源
    Result<Void> deleteDataSource(Long id);

    // 列出数据源
    Result<Page<DataSourceVO>> listDataSources(Integer page, Integer size, String keyword);
}
