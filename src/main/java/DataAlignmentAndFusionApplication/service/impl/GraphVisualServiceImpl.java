package DataAlignmentAndFusionApplication.service.impl;

import DataAlignmentAndFusionApplication.algorithm.data.GraphNode;
import DataAlignmentAndFusionApplication.service.GraphVisualService;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class GraphVisualServiceImpl implements GraphVisualService {

    // 生成首页预览图谱（简化版）
//    public EchartsConfigVO generatePreviewGraph(GraphDataVO data) {
//        EchartsConfigVO config = new EchartsConfigVO();
//
//        // 1. 节点样式规则
//        data.getNodes().forEach(node -> {
//            if (isCoreNode(node)) { // 只显示核心节点（患者/疾病/机构）
//                config.addNode(buildNodeStyle(node));
//            }
//        });
//
//        // 2. 关系样式规则
//        data.getRelations().forEach(rel -> {
//            if (isCoreRelation(rel)) {
//                config.addLink(buildLinkStyle(rel));
//            }
//        });
//
//        // 3. 添加图例配置
//        config.setLegend(buildLegend());
//
//        return config;
//    }
//
//    private boolean isCoreNode(GraphNode node) {
//        return List.of("patient", "disease", "institution")
//                .contains(node.getNodeType());
//    }
}
