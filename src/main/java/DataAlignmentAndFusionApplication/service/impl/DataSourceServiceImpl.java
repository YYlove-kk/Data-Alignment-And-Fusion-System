package DataAlignmentAndFusionApplication.service.impl;

import DataAlignmentAndFusionApplication.mapper.DataSourceMapper;
import DataAlignmentAndFusionApplication.model.DTO.DataSourceDTO;
import DataAlignmentAndFusionApplication.model.entity.DataSource;
import DataAlignmentAndFusionApplication.model.vo.DataSourceVO;
import DataAlignmentAndFusionApplication.service.DataSourceService;
import DataAlignmentAndFusionApplication.util.Result;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import org.springframework.stereotype.Service;


@Service
public class DataSourceServiceImpl
        extends ServiceImpl<DataSourceMapper, DataSource>
        implements DataSourceService {

    @Override
    public Result<DataSourceVO> createDataSource(DataSourceDTO dto) {
        return null;
    }

    @Override
    public Result<Void> deleteDataSource(Long id) {
        return null;
    }

    @Override
    public Result<Page<DataSourceVO>> listDataSources(Integer page, Integer size, String keyword) {

        return null;
    }
}