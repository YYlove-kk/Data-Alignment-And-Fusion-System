package DataAlignmentAndFusionApplication.model.vo;

import lombok.Data;

import java.time.LocalDateTime;

import lombok.Data;

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
    }

    @Data
    public static class Edge {
        private String source;
        private String target;
        private String relation;
    }
}