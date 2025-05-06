package DataAlignmentAndFusionApplication.model.vo;

import lombok.Data;

import java.util.Date;
import java.util.List;

@Data
public class GraphVO {
    private List<Node> nodes;
    private List<Edge> edges;

    @Data
    public static class Node {
        private String id;
        private String type;
        private NodeDetail nodeDetail;

        @Data
        public static class NodeDetail{
            private String textFile;
            private String imageFile;
        }

    }

    @Data
    public static class Edge {
        private String source;
        private String target;
        private List<RelationDetail> relations;

        @Data
        public static class RelationDetail {
            private String relation;  // 例如 RELATED_TO_IMAGE, CROSS_MODAL_SIMILAR
            private Double weight;    // 可以为空，例如 RELATED_TO_IMAGE 没有权重
        }
    }
}



