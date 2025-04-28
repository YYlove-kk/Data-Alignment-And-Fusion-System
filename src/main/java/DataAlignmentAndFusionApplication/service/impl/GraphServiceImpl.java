package DataAlignmentAndFusionApplication.service.impl;

import DataAlignmentAndFusionApplication.config.AppConfig;
import DataAlignmentAndFusionApplication.model.vo.GraphVO;
import DataAlignmentAndFusionApplication.service.GraphService;
import org.neo4j.driver.*;

import org.springframework.stereotype.Service;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

@Service
public class GraphServiceImpl implements GraphService {

    private static final String NEO4J_URI = "bolt://localhost:7687";
    private static final String NEO4J_USERNAME = "neo4j";
    private static final String NEO4J_PASSWORD = "12345678";

    private AppConfig appConfig;

    @Override
    public GraphVO buildKnowledgeGraph() {
        // 1. 调用 Python 脚本，导入数据
        runScript(appConfig.getInterpreterPath(), appConfig.getNeo4jScriptPath(), appConfig.getAlignOutputPath());

        // 2. 从 Neo4j 查询知识图谱
        GraphVO graphVO = new GraphVO();
        List<GraphVO.Node> nodes;
        List<GraphVO.Edge> edges;

        // 通过 Driver 创建 Session，查询数据
        try (Driver driver = GraphDatabase.driver(NEO4J_URI, AuthTokens.basic(NEO4J_USERNAME, NEO4J_PASSWORD));
             Session session = driver.session()) {

            // 查询节点
            nodes = queryNodes(session);

            // 查询关系
            edges = queryEdges(session);

        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("查询Neo4j知识图谱失败", e);
        }

        graphVO.setNodes(nodes);
        graphVO.setEdges(edges);
        return graphVO;
    }

    private void runScript(String executable, String scriptPath, String embedDirectoryPath) {
        // 组装参数
        String[] command = new String[] {
                executable,        // python 或 python3 执行命令
                scriptPath,              // python 脚本路径
                embedDirectoryPath      // 传入的目录路径
        };

        ProcessBuilder processBuilder = new ProcessBuilder(command);
        processBuilder.redirectErrorStream(true);
        try {
            // 启动 Python 进程
            Process process = processBuilder.start();
            // 输出 Python 脚本执行的日志
            BufferedReader in = new BufferedReader(new InputStreamReader(process.getInputStream()));
            String line;
            while ((line = in.readLine()) != null) {
                System.out.println(line);
            }

            // 等待脚本执行完成并检查退出码
            int exitCode = process.waitFor();
            if (exitCode != 0) {
                throw new RuntimeException("Python脚本执行失败, exitCode=" + exitCode);
            }
        } catch (IOException | InterruptedException e) {
            throw new RuntimeException("运行Python脚本时出错", e);
        }
    }

    /**
     * 查询 Neo4j 节点数据
     *
     * @param session Neo4j Session
     * @return 节点列表
     */
    private List<GraphVO.Node> queryNodes(Session session) {
        List<GraphVO.Node> nodes = new ArrayList<>();
        session.run("MATCH (n) RETURN id(n) AS id, labels(n)[0] AS label, n.id AS nodeId")
                .forEachRemaining(record -> {
                    GraphVO.Node node = new GraphVO.Node();
                    node.setId(String.valueOf(record.get("id").asInt()));
                    node.setLabel(record.get("nodeId").isNull() ? "unknown" : record.get("nodeId").asString());
                    node.setType(record.get("label").asString());
                    nodes.add(node);
                });
        return nodes;
    }

    /**
     * 查询 Neo4j 关系数据
     *
     * @param session Neo4j Session
     * @return 关系列表
     */
    private List<GraphVO.Edge> queryEdges(Session session) {
        List<GraphVO.Edge> edges = new ArrayList<>();
        session.run("MATCH (n)-[r]->(m) RETURN id(n) AS source, id(m) AS target, type(r) AS relation")
                .forEachRemaining(record -> {
                    GraphVO.Edge edge = new GraphVO.Edge();
                    edge.setSource(String.valueOf(record.get("source").asInt()));
                    edge.setTarget(String.valueOf(record.get("target").asInt()));
                    edge.setRelation(record.get("relation").asString());
                    edges.add(edge);
                });
        return edges;
    }
}