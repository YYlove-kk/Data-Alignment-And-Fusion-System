package DataAlignmentAndFusionApplication.service.impl;

import DataAlignmentAndFusionApplication.config.AppConfig;
import DataAlignmentAndFusionApplication.model.dto.GraphReq;
import DataAlignmentAndFusionApplication.model.vo.GraphVO;
import DataAlignmentAndFusionApplication.util.GraphQueryUtil;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import DataAlignmentAndFusionApplication.model.entity.FusionRecord;
import DataAlignmentAndFusionApplication.service.FusionRecordService;
import DataAlignmentAndFusionApplication.mapper.FusionRecordMapper;
import org.neo4j.driver.*;
import org.neo4j.driver.Record;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;


import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * @author 29857
 * @description 针对表【fusion_record】的数据库操作Service实现
 * @createDate 2025-04-30 21:22:01
 */
@Service
public class FusionRecordServiceImpl extends ServiceImpl<FusionRecordMapper, FusionRecord>
        implements FusionRecordService {
    @Autowired
    private FusionRecordMapper fusionRecordMapper;

    @Autowired
    private AppConfig appConfig;

    @Autowired
    private GraphQueryUtil graphQueryUtil;

    @Override
    public GraphVO fuseGraph(GraphReq req) {
        List<String> sourceIds = req.getSourceIds();
        int graphTag = req.getGraphTag();

        for (String sourceId : sourceIds) {

            if (checkFusionRecord(sourceId, graphTag)) {
                System.out.println(sourceId + " 已经与知识图谱 " + graphTag + " 融合过，跳过融合操作。");
                return null;
            }
            // 调用 Python 脚本进行融合操作
            callPythonScript(sourceId, graphTag);
            // 插入融合记录
            insertFusionRecord(sourceId, graphTag);
        }


        return graphQueryUtil.queryGraphByTag(graphTag);
    }

    /**
     * 检查 .npy 文件和知识图谱是否已经融合过
     */
    private boolean checkFusionRecord(String sourceId, int graphTag) {
        Map<String, Object> queryMap = new HashMap<>();
        queryMap.put("source_id", sourceId);
        queryMap.put("graph_tag", graphTag);
        List<FusionRecord> records = fusionRecordMapper.selectByMap(queryMap);
        return !records.isEmpty();
    }

    /**
     * 调用 Python 脚本进行融合操作
     */
    private void callPythonScript(String sourceId, int graphTag) {
        try {
            String interpreter = appConfig.getInterpreterPath(); // 比如 "python3"
            String scriptPath = appConfig.getFusionScriptPath();
            // 构建 Python 命令
            ProcessBuilder pb = new ProcessBuilder(interpreter, scriptPath, sourceId, String.valueOf(graphTag));
            // 启动进程
            Process process = pb.start();

            // 获取脚本的输出信息
            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            String line;
            while ((line = reader.readLine()) != null) {
                System.out.println(line);
            }

            // 等待脚本执行完成
            int exitCode = process.waitFor();
            if (exitCode == 0) {
                System.out.println("Python 脚本执行成功");
            } else {
                System.out.println("Python 脚本执行失败，退出码: " + exitCode);
            }
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
    }

    /**
     * 将 .npy 文件和知识图谱的融合记录插入到 MySQL 表中
     */
    private void insertFusionRecord(String sourceId, int graphTag) {
        FusionRecord record = new FusionRecord();
        record.setSourceId(sourceId);
        record.setGraphTag(graphTag);
        fusionRecordMapper.insert(record);
    }

    @Override
    public List<Integer> getAvailableGraph() {
        List<Integer> tagList = new ArrayList<>();
        try (Driver driver = GraphDatabase.driver("bolt://localhost:7687",
                AuthTokens.basic("neo4j", "12345678"));
             Session session = driver.session()) {
            String cypher = """
                        MATCH (n)
                        WHERE exists(n.tag)
                        RETURN DISTINCT n.tag AS tag
                        ORDER BY tag
                    """;

            for (Result it = session.run(cypher); it.hasNext(); ) {
                Record record = it.next();
                if (!record.get("tag").isNull()) {
                    tagList.add(record.get("tag").asInt());
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("查询 Neo4j 中可用图谱 tag 失败", e);
        }

        return tagList;
    }

}




