package DataAlignmentAndFusionApplication.service.impl;

import DataAlignmentAndFusionApplication.config.AppConfig;
import DataAlignmentAndFusionApplication.service.DimReduceService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.util.List;

@Service
public class DimReduceServiceImpl implements DimReduceService {

    @Autowired
    private AppConfig appConfig;

    @Override
    public void reduce() {
        try {
            // 构建Python脚本命令，传递参数
            String interpreter = appConfig.getInterpreterPath();
            String scriptPath = appConfig.getKpcaReducePath();
            String inDir = appConfig.getAlignOutputPath();
            String outDir = appConfig.getAlignOutputPath() + "reduce/";

            String command = String.format("%s %s --input_dir %s --output_dir %s", interpreter, scriptPath, inDir, outDir);

            // 执行 Python 脚本
            Process process = Runtime.getRuntime().exec(command);

            // 等待脚本执行完成
            int exitCode = process.waitFor();
            if (exitCode == 0) {
                System.out.println("Dimension reduction completed successfully.");
            } else {
                throw new RuntimeException("Dimension reduction failed with exit code: " + exitCode);
            }
        } catch (IOException | InterruptedException e) {
            // 异常处理
            throw new RuntimeException("Error occurred while executing the Python script: " + e.getMessage(), e);
        }
    }
}

