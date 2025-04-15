package DataAlignmentAndFusionApplication.mapper.module;

import DataAlignmentAndFusionApplication.model.entity.AlignmentDetail;
import DataAlignmentAndFusionApplication.model.entity.AlignmentResult;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

import java.util.List;

@Mapper
public interface AlignmentResultMapper {
    // 批量插入对齐详情（需在XML中实现）
    void insertBatchDetails(@Param("list") List<AlignmentDetail> details);

    // 分页查询（使用MyBatis-Plus原生分页）
    Page<AlignmentResult> selectPageByConfig(Page<AlignmentResult> page, @Param("configId") Long configId);
}
