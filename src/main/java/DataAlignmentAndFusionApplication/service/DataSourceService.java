package DataAlignmentAndFusionApplication.service;

import DataAlignmentAndFusionApplication.model.DTO.DataSourceDTO;
import DataAlignmentAndFusionApplication.model.vo.DataSourceVO;
import DataAlignmentAndFusionApplication.util.Result;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface DataSourceService {

    Result<DataSourceVO> createDataSource(DataSourceDTO dto);

    Result<Void> deleteDataSource(Long id);

    Result<Page<DataSourceVO>> listDataSources(Integer page, Integer size, String keyword);
}