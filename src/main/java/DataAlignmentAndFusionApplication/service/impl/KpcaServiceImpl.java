package DataAlignmentAndFusionApplication.service.impl;

import DataAlignmentAndFusionApplication.config.AppConfig;
import DataAlignmentAndFusionApplication.model.dto.GraphReq;
import DataAlignmentAndFusionApplication.service.KpcaService;
import DataAlignmentAndFusionApplication.util.Result;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.List;

@Service
public class KpcaServiceImpl implements KpcaService {
    @Autowired
    private AppConfig appConfig;

    @Override
    public Result<String> runKpcaReduction(GraphReq req) {

        List<String> sourceIds = req.getSourceIds();

        StringBuilder summary = new StringBuilder();

        for (String sourceId : sourceIds) {
            try {
                ProcessBuilder builder = new ProcessBuilder(
                        appConfig.getInterpreterPath(),
                        appConfig.getKpcaReducePath()
                );
                builder.redirectErrorStream(true);

                Process process = builder.start();
                BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
                StringBuilder outputInfo = new StringBuilder();
                String line;
                while ((line = reader.readLine()) != null) {
                    outputInfo.append(line).append("\n");
                }

                int exitCode = process.waitFor();
                if (exitCode == 0) {
                    summary.append("Source ").append(sourceId).append(": Success\n");
                } else {
                    summary.append("Source ").append(sourceId).append(": Failed\n").append(outputInfo);
                }

            } catch (Exception e) {
                summary.append("Source ").append(sourceId).append(": Exception occurred - ").append(e.getMessage()).append("\n");
                e.printStackTrace(); // 记录日志或用于调试
            }
        }

        return Result.success("KPCA 执行完成。\n" + summary);
    }
}