package DataAlignmentAndFusionApplication.service.impl;

import DataAlignmentAndFusionApplication.common.TagGenerator;
import DataAlignmentAndFusionApplication.config.AppConfig;
import DataAlignmentAndFusionApplication.mapper.EmbedRecordMapper;
import DataAlignmentAndFusionApplication.mapper.ReduceRecordMapper;
import DataAlignmentAndFusionApplication.mapper.UploadRecordMapper;
import DataAlignmentAndFusionApplication.model.dto.DeleteReq;
import DataAlignmentAndFusionApplication.model.dto.GraphReq;
import DataAlignmentAndFusionApplication.model.entity.BuildRecord;
import DataAlignmentAndFusionApplication.model.entity.ReduceRecord;
import DataAlignmentAndFusionApplication.model.entity.UploadRecord;
import DataAlignmentAndFusionApplication.model.vo.GraphVO;
import DataAlignmentAndFusionApplication.service.BuildRecordService;
import DataAlignmentAndFusionApplication.service.GraphService;
import DataAlignmentAndFusionApplication.util.GraphQueryUtil;

import DataAlignmentAndFusionApplication.util.Result;
import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import org.neo4j.driver.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;


@Service
public class GraphServiceImpl implements GraphService {

    @Autowired
    private AppConfig appConfig;

    @Autowired
    private TagGenerator tagGenerator;

    @Autowired
    private GraphQueryUtil graphQueryUtil;

    @Autowired
    private BuildRecordService buildRecordService;

    @Autowired
    private EmbedRecordMapper embedRecordMapper;

    @Autowired
    private UploadRecordMapper uploadRecordMapper;

    @Autowired
    private ReduceRecordMapper reduceRecordMapper;


    @Override
    public GraphVO buildKnowledgeGraph(GraphReq req) {

        List<String> sourceIds = req.getSourceIds();

        int mode = req.getMode();
        // 生成唯一的 tag 值
        int tag = tagGenerator.generateTag();

        for (String sourceId : sourceIds) {

//                Set<String> patientIds = getAllDistinctPatientIds(sourceId);
//                String patientIdsStr = String.join(",", patientIds);
            List<String> patientIds = reduceRecordMapper.selectList(new QueryWrapper<ReduceRecord>().eq("source_id", sourceId))
                    .stream().map(ReduceRecord::getPatientId)
                    .toList();
            String patientIdsStr = String.join(",", patientIds);

            UploadRecord r = uploadRecordMapper.selectOne(new QueryWrapper<UploadRecord>().eq("source_id", sourceId));
            String institution = r.getInstitution();
            String type = r.getModalityType();

            runImportScript(appConfig.getInterpreterPath(), appConfig.getNeo4jScriptPath(), patientIdsStr, tag, institution, type);
            runBuildScript(appConfig.getInterpreterPath(), appConfig.getNeo4jScriptPath(), patientIdsStr,mode, tag);

            BuildRecord record = new BuildRecord();
            record.setSourceId(sourceId);
            record.setGraphTag(tag);
            record.setMode(mode);
            buildRecordService.save(record);
        }

        return graphQueryUtil.queryGraphByTag(tag);
    }

    @Override
    public Result<String> deleteEdge(DeleteReq req) {

        try (Driver driver = GraphDatabase.driver("bolt://localhost:7687",
                AuthTokens.basic("neo4j", "12345678"));
             Session session = driver.session()) {
            String cypher = String.format("""
                    MATCH (a)-[r:%s {tag: $tag}]->(b)
                    WHERE
                      (a:Text AND b:Image AND a.uuid = $uuid1 AND b.uuid = $uuid2)
                      OR
                      (a:Image AND b:Text AND a.uuid = $uuid1 AND b.uuid = $uuid2)
                    DELETE r
                    """, req.getEdgeType());

            session.writeTransaction(tx -> tx.run(cypher,
                    Values.parameters(
                            "uuid1", req.getUuid1(),
                            "uuid2", req.getUuid2(),
                            "tag", req.getTag()
                    )
            ));
            return Result.success("Edge deleted successfully.");
        } catch (Exception e) {
            return Result.error(500, "Failed to delete edge: " + e.getMessage());
        }
    }

    @Override
    public GraphVO getGraph(int tag) {
        return graphQueryUtil.queryGraphByTag(tag);
    }

//    public Set<String> getAllDistinctPatientIds(String sourceId) {
//        Set<String> resultSet = new HashSet<>();
//        List<String> patientIds = embedRecordMapper.selectDistinctPatientIdsBySourceId(sourceId);
//        resultSet.addAll(patientIds); // 去重
//
//        return resultSet;
//    }



    private void runImportScript(String executable, String scriptPath, String patientIdsStr, int tag, String institution,String type) {
        // 组装参数
        String[] command = new String[]{
                executable,
                scriptPath,
                patientIdsStr,
                String.valueOf(tag),
                institution,
                type
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

    private void runBuildScript(String executable, String scriptPath, String patientIdsStr, int tag, int mode) {
        // 组装参数
        String[] command = new String[]{
                executable,
                scriptPath,
                patientIdsStr,
                String.valueOf(tag),
                String.valueOf(mode)

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

}