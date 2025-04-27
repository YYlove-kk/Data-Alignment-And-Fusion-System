package DataAlignmentAndFusionApplication.service.impl;

import DataAlignmentAndFusionApplication.config.AppConfig;
import DataAlignmentAndFusionApplication.mapper.module.AlignmentResultMapper;
import DataAlignmentAndFusionApplication.model.entity.AlignmentResult;
import DataAlignmentAndFusionApplication.service.AlignmentService;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import org.json.JSONArray;
import org.json.JSONObject;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.List;

@Slf4j
@Service
public class AlignmentServiceImpl implements AlignmentService {

    @Autowired
    private AppConfig appConfig;


    @Autowired
    private AlignmentResultMapper alignmentResultMapper;

    @Override
    public AlignmentResult alignTextAndImage() throws Exception {
        // 构造参数列表
        String dataDir = appConfig.getUploadOutputDir();
        List<String> arguments = List.of(
                "--txt_dir", dataDir+"text/",
                "--img_dir", dataDir+"image/",
                "--output_dir", appConfig.getAlignOutputPath()
        );

        // 调用 Python 脚本并获取结果
        String result = runPythonScript(appConfig.getAlignScriptPath(), arguments);


        // 解析 Python 返回的 JSON 数据
        JSONObject json = new JSONObject(result);
        JSONArray alignmentMatrix = json.getJSONArray("alignment_matrix");
        double semanticAccuracy = json.getDouble("semantic_accuracy");
        int alignmentCoverage = json.getInt("alignment_coverage");

        // 将对齐结果转换为 AlignmentResult 对象
        AlignmentResult alignmentResult = new AlignmentResult();
        // 使用 ObjectMapper 将 JSONArray 转换为 JSON 字符串
        ObjectMapper objectMapper = new ObjectMapper();
        String alignmentMatrixJsonString = objectMapper.writeValueAsString(alignmentMatrix);
        alignmentResult.setAlignmentMatrix(alignmentMatrixJsonString);  // 将其保存为字符串

        alignmentResult.setSemanticAccuracy(semanticAccuracy);
        alignmentResult.setAlignmentCoverage(alignmentCoverage);

        // 保存到数据库
        alignmentResultMapper.insert(alignmentResult);

        return alignmentResult;
    }

    private String runPythonScript(String scriptPath, List<String> arguments) throws IOException, InterruptedException {
        // 设置 Python 解释器和脚本路径
        ProcessBuilder processBuilder = new ProcessBuilder();
        processBuilder.command("python", scriptPath);

        // 将参数传递给脚本
        processBuilder.command().addAll(arguments);

        // 启动进程并等待执行完成
        Process process = processBuilder.start();

        // 读取脚本的标准输出
        BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
        StringBuilder output = new StringBuilder();
        String line;
        while ((line = reader.readLine()) != null) {
            output.append(line).append("\n");
        }

        // 获取脚本的退出值
        int exitCode = process.waitFor();
        if (exitCode != 0) {
            throw new IOException("Python script execution failed with exit code " + exitCode);
        }

        return output.toString();
    }

    public double[][] convertJsonArrayToDoubleArray(JSONArray jsonArray) {
        int rows = jsonArray.length();
        int cols = jsonArray.getJSONArray(0).length();
        double[][] matrix = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            JSONArray row = jsonArray.getJSONArray(i);
            for (int j = 0; j < cols; j++) {
                matrix[i][j] = row.getDouble(j);
            }
        }
        return matrix;
    }
}

