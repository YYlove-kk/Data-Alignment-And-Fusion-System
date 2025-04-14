package DataAlignmentAndFusionApplication.algorithm.data;

import lombok.Data;
import org.w3c.dom.Node;

import javax.management.relation.Relation;
import java.util.ArrayList;
import java.util.List;

//图谱数据
@Data
public class GraphData {
    private List<GraphNode> nodes = new ArrayList<>();
    private List<GraphRelation> relations = new ArrayList<>();

    public void addNode(GraphNode node) {
        nodes.add(node);
    }

    public void addRelation(GraphRelation relation) {
        relations.add(relation);
    }
}