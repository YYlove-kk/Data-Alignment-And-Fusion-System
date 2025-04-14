package DataAlignmentAndFusionApplication.mapper;

import DataAlignmentAndFusionApplication.model.entity.KgFusionRecord;
import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.util.List;

public interface KgFusionRecordMapper extends BaseMapper<KgFusionRecord> {

    // 查询主图谱的所有融合记录
//    @Select("SELECT * FROM kg_fusion_record WHERE main_graph_id = #{mainGraphId} ORDER BY create_time DESC")
    List<KgFusionRecord> selectByMainGraphId(@Param("mainGraphId") Long mainGraphId);

    // 查询子图谱的融合去向
//    @Select("SELECT * FROM kg_fusion_record WHERE sub_graph_id = #{subGraphId}")
    KgFusionRecord selectBySubGraphId(@Param("subGraphId") Long subGraphId);

    // 统计某策略的使用次数
//    @Select("SELECT COUNT(*) FROM kg_fusion_record WHERE strategy_type = #{strategy}")
    int countByStrategy(@Param("strategy") String strategy);
}
