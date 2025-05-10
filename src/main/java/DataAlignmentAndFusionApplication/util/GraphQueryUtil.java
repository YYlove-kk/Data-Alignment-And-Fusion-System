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
                RETURN id(n) AS id, labels(n)[0] AS type, coalesce(n.id, n.uuid, 'unknown') AS label
                """;

            Map<String, Object> params = tag != null ? Map.of("tag", tag) : Map.of();

            session.run(nodeCypher, params)
                    .forEachRemaining(record -> {
                        GraphVO.Node node = new GraphVO.Node();
                        node.setId(String.valueOf(record.get("id").asInt()));
                        node.setType(record.get("label").isNull() ? "unknown" : record.get("label").asString());

                        if ("Patient".equals(node.getType())) {
                            String patientId = node.getId();
                            JointEmbeddingRelation jointEmbeddingRelation = jointEmbeddingRelationMapper.selectOne(
                                    new QueryWrapper<JointEmbeddingRelation>().eq("patient_id", patientId));
                            if (jointEmbeddingRelation != null) {
                                GraphVO.Node.NodeDetail d = new GraphVO.Node.NodeDetail();
                                d.setTextFile(jointEmbeddingRelation.getTextFile());
                                d.setImageFile(jointEmbeddingRelation.getImageFile());
                                node.setNodeDetail(d);
                            }
                        }
                        nodes.add(node);
                    });

            // 构造动态 Cypher 查询（关系）
            String edgeCypher = """
                MATCH (n)-[r]->(m)
                """ + (tag != null ? "WHERE n.tag = $tag AND m.tag = $tag AND r.tag = $tag\n" : "") + """
                RETURN id(n) AS source, id(m) AS target, type(r) AS relation, r.weight AS weight
                """;

            session.run(edgeCypher, params)
                    .forEachRemaining(record -> {
                        String source = String.valueOf(record.get("source").asInt());
                        String target = String.valueOf(record.get("target").asInt());
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
