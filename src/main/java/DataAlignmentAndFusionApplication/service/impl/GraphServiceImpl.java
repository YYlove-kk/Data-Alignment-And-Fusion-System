package DataAlignmentAndFusionApplication.service.impl;

import DataAlignmentAndFusionApplication.config.AppConfig;
import DataAlignmentAndFusionApplication.mapper.UploadRecordMapper;
import DataAlignmentAndFusionApplication.model.dto.GraphReq;
import DataAlignmentAndFusionApplication.model.vo.GraphVO;
import DataAlignmentAndFusionApplication.service.GraphService;
import DataAlignmentAndFusionApplication.util.GraphQueryUtil;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.concurrent.atomic.AtomicInteger;

@Service
public class GraphServiceImpl implements GraphService {

    @Autowired
    private AppConfig appConfig;

    private static final AtomicInteger tagCounter = new AtomicInteger(1);

    @Autowired
    private GraphQueryUtil graphQueryUtil;

    @Override
    public GraphVO buildKnowledgeGraph(GraphReq req) {

        String patientId = req.getPatientId();
        int mode = req.getMode().getCode();
        // 生成唯一的 tag 值
        int tag = generateTag();

        // 1. 调用 Python 脚本，导入数据
        runScript(appConfig.getInterpreterPath(), appConfig.getNeo4jScriptPath(), patientId, tag, mode);

        // 调用 hnsw_builder.py
        runScript(appConfig.getInterpreterPath(), appConfig.getKnswScriptPath(), patientId, tag, mode);

        // 2. 从 Neo4j 查询知识图谱
        return graphQueryUtil.queryGraphByTag(tag);
    }

    private int generateTag() {
        // 原子自增
        return tagCounter.getAndIncrement();
    }

    private void runScript(String executable, String scriptPath, String patientId, int tag, int mode) {
        // 组装参数
        String[] command = new String[]{
                executable,
                scriptPath,
                patientId,
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