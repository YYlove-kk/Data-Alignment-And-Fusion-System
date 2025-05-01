package DataAlignmentAndFusionApplication.util;

import DataAlignmentAndFusionApplication.mapper.UploadRecordMapper;
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
    private UploadRecordMapper uploadRecordMapper;

    public GraphVO queryGraphByTag(int tag) {
        List<GraphVO.Node> nodes = new ArrayList<>();
        Map<String, GraphVO.Edge> edgeMap = new HashMap<>();

        try (Driver driver = GraphDatabase.driver(uri, AuthTokens.basic(username, password));
             Session session = driver.session()) {

            // 查询节点

            String nodeCypher = """
                    MATCH (n)
                    WHERE n.tag = $tag
                    RETURN id(n) AS id, labels(n)[0] AS type, coalesce(n.id, n.uuid, 'unknown') AS label
                    """;

            session.run(nodeCypher, Map.of("tag", tag))
                    .forEachRemaining(record -> {
                        GraphVO.Node node = new GraphVO.Node();
                        node.setId(String.valueOf(record.get("id").asInt()));
                        node.setName(record.get("label").isNull() ? "unknown" : record.get("label").asString());
                        node.setType(record.get("type").asString());

                        if ("Patient".equals(node.getType())) {
                            String patientId = node.getName();
                            List<UploadRecord> uploadRecords = uploadRecordMapper.selectList(
                                    new QueryWrapper<UploadRecord>().eq("patient_id", patientId));
                            if (!uploadRecords.isEmpty()) {
                                List<GraphVO.Node.NodeDetail> nodeDetails = uploadRecords.stream().map(ur -> {
                                    GraphVO.Node.NodeDetail d = new GraphVO.Node.NodeDetail();
                                    d.setFileName(ur.getFileName());
                                    d.setModalityType(ur.getModalityType());
                                    d.setInstitution(ur.getInstitution());
                                    d.setProcessTime(ur.getProcessTime());
                                    return d;
                                }).collect(Collectors.toList());
                                node.setNodeDetail(nodeDetails);
                            }
                        }
                        nodes.add(node);
                    });

            // 查询关系
            String edgeCypher = """
                    MATCH (n)-[r]->(m)
                    WHERE n.tag = $tag AND m.tag = $tag AND r.tag = $tag
                    RETURN id(n) AS source, id(m) AS target, type(r) AS relation, r.weight AS weight
                    """;

            session.run(edgeCypher, Map.of("tag", tag))
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
        graphVO.setEdges(new ArrayList<>(edgeMap.values()));  // 转换为 List
        return graphVO;
    }

}
