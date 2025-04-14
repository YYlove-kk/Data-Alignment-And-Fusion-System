package DataAlignmentAndFusionApplication.mapper;

import DataAlignmentAndFusionApplication.model.entity.KgConstructionRecord;
import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;


public interface KgConstructionRecordMapper extends BaseMapper<KgConstructionRecord> {

    // 按用户分页查询构建记录
//    @Select("SELECT * FROM kg_construction_record WHERE user_id = #{userId} ORDER BY create_time DESC")
    Page<KgConstructionRecord> selectByUserId(Page<KgConstructionRecord> page, @Param("userId") Long userId);

    // 检查某数据集是否已被用于构建
//    @Select("SELECT COUNT(*) FROM kg_construction_record WHERE dataset_id = #{datasetId}")
    int countByDatasetId(@Param("datasetId") String datasetId);
}