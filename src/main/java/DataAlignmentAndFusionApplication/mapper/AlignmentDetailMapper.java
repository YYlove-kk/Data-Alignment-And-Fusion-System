package DataAlignmentAndFusionApplication.mapper;

import DataAlignmentAndFusionApplication.model.entity.AlignmentDetail;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

import java.util.List;

@Mapper
public interface AlignmentDetailMapper {
    // 根据结果ID查询详情列表
    List<AlignmentDetail> selectByResultId(@Param("resultId") Long resultId);
}
