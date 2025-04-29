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
        private String label;
        private String type;
        private List<NodeDetail> nodeDetail;

        @Data
        public static class NodeDetail{
            private String fileName;
            private String modalityType;
            private String institution;
            private Date processTime;
        }

    }

    @Data
    public static class Edge {
        private String source;
        private String target;
        private String relation;
        private double weight;
    }
}



