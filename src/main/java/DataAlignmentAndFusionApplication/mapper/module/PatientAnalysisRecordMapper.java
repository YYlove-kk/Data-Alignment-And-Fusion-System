package DataAlignmentAndFusionApplication.mapper.module;

import DataAlignmentAndFusionApplication.model.entity.PatientAnalysisRecord;
import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import org.apache.ibatis.annotations.Param;

import java.util.List;
import java.util.Map;

public interface PatientAnalysisRecordMapper extends BaseMapper<PatientAnalysisRecord> {

    // 按用户查询历史记录
//    @Select("SELECT * FROM patient_analysis_record WHERE user_id = #{userId} ORDER BY create_time DESC")
    Page<PatientAnalysisRecord> selectByUser(Page<?> page, @Param("userId") Long userId);

    // 按类型统计使用次数
//    @Select("SELECT analysis_type, COUNT(*) as count FROM patient_analysis_record " +
//            "WHERE user_id = #{userId} GROUP BY analysis_type")
    List<Map<String, Object>> countByAnalysisType(@Param("userId") Long userId);
}