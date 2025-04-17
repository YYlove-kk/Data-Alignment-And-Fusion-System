package generator.service.impl;

import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import DataAlignmentAndFusionApplication.model.entity.DataSource;
import generator.service.DataSourceService;
import DataAlignmentAndFusionApplication.mapper.module.DataSourceMapper;
import org.springframework.stereotype.Service;

/**
* @author 29857
* @description 针对表【data_source(数据源管理表)】的数据库操作Service实现
* @createDate 2025-04-17 20:02:26
*/
@Service
public class DataSourceServiceImpl extends ServiceImpl<DataSourceMapper, DataSource>
    implements DataSourceService{

}




