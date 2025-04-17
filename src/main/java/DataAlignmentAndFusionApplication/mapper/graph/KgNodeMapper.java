package DataAlignmentAndFusionApplication.mapper.graph;

import DataAlignmentAndFusionApplication.model.entity.KgNode;
import com.baomidou.mybatisplus.core.mapper.BaseMapper;

public interface KgNodeMapper extends BaseMapper<KgNode> {

    // 批量插入节点（MyBatis-Plus的insertBatch在某些版本需自定义）
//    @Insert("<script>" +
//            "INSERT INTO kg_node_detail (node_id, graph_id, properties) VALUES " +
//            "<foreach collection='list' item='item' separator=','>" +
//            "(#{item.nodeId}, #{item.graphId}, #{item.properties})" +
//            "</foreach>" +
//            "</script>")
//    int insertBatch(@Param("list") List<KgNodeDetail> nodes);
}