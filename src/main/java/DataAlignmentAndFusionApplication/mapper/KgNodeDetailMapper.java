package DataAlignmentAndFusionApplication.mapper;

import DataAlignmentAndFusionApplication.model.entity.KgNodeDetail;
import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import org.apache.ibatis.annotations.Insert;
import org.apache.ibatis.annotations.Param;

public interface KgNodeDetailMapper extends BaseMapper<KgNodeDetail> {

    // 批量插入节点（MyBatis-Plus的insertBatch在某些版本需自定义）
//    @Insert("<script>" +
//            "INSERT INTO kg_node_detail (node_id, graph_id, properties) VALUES " +
//            "<foreach collection='list' item='item' separator=','>" +
//            "(#{item.nodeId}, #{item.graphId}, #{item.properties})" +
//            "</foreach>" +
//            "</script>")
//    int insertBatch(@Param("list") List<KgNodeDetail> nodes);
}