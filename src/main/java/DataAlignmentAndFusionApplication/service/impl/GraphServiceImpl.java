package DataAlignmentAndFusionApplication.service.impl;

import DataAlignmentAndFusionApplication.config.AppConfig;
import DataAlignmentAndFusionApplication.mapper.EmbedRecordMapper;
import DataAlignmentAndFusionApplication.model.dto.GraphReq;
import DataAlignmentAndFusionApplication.model.entity.BuildRecord;
import DataAlignmentAndFusionApplication.model.vo.GraphVO;
import DataAlignmentAndFusionApplication.service.BuildRecordService;
import DataAlignmentAndFusionApplication.service.GraphService;
import DataAlignmentAndFusionApplication.util.GraphQueryUtil;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

@Service
public class GraphServiceImpl implements GraphService {

    @Autowired
    private AppConfig appConfig;

    private static final AtomicInteger tagCounter = new AtomicInteger(1);

    @Autowired
    private GraphQueryUtil graphQueryUtil;

    @Autowired
    private BuildRecordService buildRecordService;

    @Autowired
    private EmbedRecordMapper embedRecordMapper;


    @Override
    public GraphVO buildKnowledgeGraph(GraphReq req) {

        List<String> sourceIds = req.getSourceIds();

        int mode = req.getMode().getCode();
        // 生成唯一的 tag 值
        int tag = generateTag();



        if (mode == 0) {
            for (String sourceId : sourceIds) {

                Set<String> patientIds = getAllDistinctPatientIds(sourceId);
                String patientIdsStr = String.join(",", patientIds);

                runScript(appConfig.getInterpreterPath(), appConfig.getNeo4jScriptPath(), patientIdsStr, tag);
                BuildRecord record = new BuildRecord();
                record.setSourceId(sourceId);
                record.setGraphTag(tag);
                record.setMode(mode);
                buildRecordService.save(record);
            }


            // 2. 从 Neo4j 查询知识图谱
        }
        return graphQueryUtil.queryGraphByTag(tag);
    }

    public Set<String> getAllDistinctPatientIds(String sourceId) {
        Set<String> resultSet = new HashSet<>();
            List<String> patientIds = embedRecordMapper.selectDistinctPatientIdsBySourceId(sourceId);
            resultSet.addAll(patientIds); // 去重

        return resultSet;
    }

    private int generateTag() {
        // 原子自增
        return tagCounter.getAndIncrement();
    }

    private void runScript(String executable, String scriptPath, String patientIds, int tag) {
        // 组装参数
        String[] command = new String[]{
                executable,
                scriptPath,
                patientIds,
                String.valueOf(tag),
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