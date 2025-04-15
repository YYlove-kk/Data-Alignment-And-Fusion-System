package DataAlignmentAndFusionApplication.mapper.module;

import DataAlignmentAndFusionApplication.model.entity.AlignmentConfig;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

@Mapper
public interface AlignmentConfigMapper {
    // 查询用户最新配置
    AlignmentConfig selectLatestByUser(@Param("userId") Long userId);
}
