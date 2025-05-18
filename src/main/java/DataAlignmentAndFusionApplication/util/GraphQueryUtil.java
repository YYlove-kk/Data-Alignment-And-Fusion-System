package DataAlignmentAndFusionApplication.util;

import DataAlignmentAndFusionApplication.mapper.JointEmbeddingRelationMapper;
import DataAlignmentAndFusionApplication.mapper.UploadRecordMapper;
import DataAlignmentAndFusionApplication.model.entity.JointEmbeddingRelation;
import DataAlignmentAndFusionApplication.model.entity.UploadRecord;
import DataAlignmentAndFusionApplication.model.vo.GraphVO;
import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import org.neo4j.driver.AuthTokens;
import org.neo4j.driver.Driver;
import org.neo4j.driver.GraphDatabase;
import org.neo4j.driver.Session;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@Component
public class GraphQueryUtil {
    @Value("bolt://localhost:7687")
    private String uri;

    @Value("neo4j")
    private String username;

    @Value("12345678")
    private String password;

    @Autowired
    private JointEmbeddingRelationMapper jointEmbeddingRelationMapper;

    public GraphVO queryGraphByTag(Integer tag) {
        List<GraphVO.Node> nodes = new ArrayList<>();
        Map<String, GraphVO.Edge> edgeMap = new HashMap<>();

        try (Driver driver = GraphDatabase.driver(uri, AuthTokens.basic(username, password));
             Session session = driver.session()) {

            // 构造动态 Cypher 查询（节点）
            String nodeCypher = """
                MATCH (n)
                """ + (tag != null ? "WHERE n.tag = $tag\n" : "") + """
                RETURN id(n) AS id, labels(n)[0] AS type,  n.name AS name
                """;


            Map<String, Object> params = tag != null ? Map.of("tag", tag) : Map.of();

            session.run(nodeCypher, params)
                    .forEachRemaining(record -> {
                        GraphVO.Node node = new GraphVO.Node();
                        node.setType(record.get("type").asString());
                        node.setId(record.get("name").asString());
                        GraphVO.Node.NodeDetail d = new GraphVO.Node.NodeDetail();
                        if ("Patient".equals(node.getType())) {

                                d.setTextFile(node.getId()+"_z_t.npy");
                                d.setImageFile(node.getId()+"_z_i.npy");
                                node.setNodeDetail(d);
                        }
                        if ("Text".equals(node.getType())) {
                            d.setTextFile(node.getId()+".npy");
                        }
                        if ("Image".equals(node.getType())) {
                            d.setImageFile(node.getId()+".npy");
                        }
                        nodes.add(node);
                    });

            // 构造动态 Cypher 查询（关系）
            String edgeCypher = """
                MATCH (n)-[r]->(m)
                """ + (tag != null ? "WHERE n.tag = $tag AND m.tag = $tag AND r.tag = $tag\n" : "") + """
                RETURN n.name AS source, m.name AS target, type(r) AS relation, r.weight AS weight
                """;

            session.run(edgeCypher, params)
                    .forEachRemaining(record -> {
                        String source = record.get("source").asString();
                        String target = record.get("target").asString();
                        String key = source + "->" + target;

                        GraphVO.Edge.RelationDetail detail = new GraphVO.Edge.RelationDetail();
                        detail.setRelation(record.get("relation").asString());
                        if (!record.get("weight").isNull()) {
                            detail.setWeight(record.get("weight").asDouble());
                        }

                        edgeMap.computeIfAbsent(key, k -> {
                            GraphVO.Edge edge = new GraphVO.Edge();
                            edge.setSource(source);
                            edge.setTarget(target);
                            edge.setRelations(new ArrayList<>());
                            return edge;
                        }).getRelations().add(detail);
                    });

        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("查询 Neo4j 知识图谱失败", e);
        }

        GraphVO graphVO = new GraphVO();
        graphVO.setNodes(nodes);
        graphVO.setEdges(new ArrayList<>(edgeMap.values()));
        return graphVO;
    }

}
