package DataAlignmentAndFusionApplication.service.impl;

import DataAlignmentAndFusionApplication.config.AppConfig;
import DataAlignmentAndFusionApplication.mapper.AlignmentResultMapper;
import DataAlignmentAndFusionApplication.mapper.JointEmbeddingRelationMapper;
import DataAlignmentAndFusionApplication.mapper.UploadRecordMapper;
import DataAlignmentAndFusionApplication.model.entity.AlignmentResult;
import DataAlignmentAndFusionApplication.model.entity.JointEmbeddingRelation;
import DataAlignmentAndFusionApplication.model.vo.AlignmentVO;
import DataAlignmentAndFusionApplication.service.AlignmentService;
import DataAlignmentAndFusionApplication.util.CsvImporter;
import DataAlignmentAndFusionApplication.util.Result;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import org.json.JSONArray;
import org.json.JSONObject;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import com.fasterxml.jackson.core.type.TypeReference;


import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;
import java.util.stream.Collectors;

@Slf4j
@Service
public class AlignmentServiceImpl extends ServiceImpl<AlignmentResultMapper, AlignmentResult> implements AlignmentService {

    @Autowired
    private AppConfig appConfig;

    @Autowired
    private CsvImporter csvImporter;


    @Autowired
    private JointEmbeddingRelationMapper jointEmbeddingRelationMapper;

    @Override
    public Result<String> alignTextAndImage() throws Exception {
        try {
            // 构造参数列表
            String dataDir = appConfig.getAlignSourcePath();
            List<String> arguments = List.of(
                    "--txt_dir", dataDir + "text",
                    "--img_dir", dataDir + "image",
                    "--output_dir", appConfig.getAlignOutputPath()
            );

            // 调用 Python 脚本并获取结果
            String result = runPythonScript(appConfig.getAlignScriptPath(), arguments);

            // 解析 Python 返回的 JSON 数据
            JSONObject json = new JSONObject(result);
            JSONArray alignmentMatrix = json.getJSONArray("alignment_matrix");
            JSONArray diagonalSimilarity = json.getJSONArray("diagonal_similarity");
            double semanticAccuracy = json.getDouble("semantic_accuracy");
            int alignmentCoverage = json.getInt("coverage");

            JSONArray alignedFilenames = json.getJSONArray("aligned_filenames");
            List<String> filenamePairs = new ArrayList<>();

            for (int i = 0; i < alignedFilenames.length(); i++) {
                JSONObject pair = alignedFilenames.getJSONObject(i);
                String textFile = pair.getString("text_file");
                String imageFile = pair.getString("image_file");

                // 存成格式化字符串，可以根据需要调整格式
                String formatted = String.format("Text: %s, Image: %s", textFile, imageFile);
                filenamePairs.add(formatted);
            }

            // 构造 AlignmentResult 对象
            AlignmentResult alignmentResult = new AlignmentResult();
            ObjectMapper objectMapper = new ObjectMapper();
            alignmentResult.setAlignmentMatrix(objectMapper.writeValueAsString(alignmentMatrix));
            alignmentResult.setSemanticAccuracy(semanticAccuracy);
            alignmentResult.setAlignmentCoverage(alignmentCoverage);
            alignmentResult.setDiagonalSimilarity(diagonalSimilarity.toString());
            alignmentResult.setFilenamePairs(filenamePairs);

            // 保存结果到数据库
            save(alignmentResult);

            // 返回封装的成功结果
            return Result.success("对齐成功");
        } catch (Exception e) {
            e.printStackTrace(); // 可选：记录日志
            return Result.error(500, "对齐失败: " + e.getMessage());
        }
    }


    @Override
    public List<AlignmentVO> getAllResults() {
        List<AlignmentResult> resultList = list();
        List<AlignmentVO> alignmentVOList = new ArrayList<>();
        for (AlignmentResult result : resultList) {
            AlignmentVO vo = new AlignmentVO();

            vo.setFilenamePairs(result.getFilenamePairs());
            vo.setAccuracy(result.getSemanticAccuracy());
            vo.setCoverage(String.valueOf(result.getAlignmentCoverage())); // int 转为 String

            alignmentVOList.add(vo);
        }

        return alignmentVOList;
    }

    @Override
    public List<String> getPatients() {
        // 使用 LambdaQueryWrapper 去重查询所有 patient_id
        return jointEmbeddingRelationMapper.selectList(
                        new LambdaQueryWrapper<JointEmbeddingRelation>()
                                .select(JointEmbeddingRelation::getPatientId)
                                .groupBy(JointEmbeddingRelation::getPatientId)
                ).stream()
                .map(JointEmbeddingRelation::getPatientId)
                .collect(Collectors.toList());
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
        while ((line = reader.readLine()) != null) {
            if (line.startsWith("[CSV_PATH]")) {
                String csvPath = line.substring("[CSV_PATH]".length());
                csvImporter.importFromCsv(csvPath);
            }
        }
        // 获取脚本的退出值
        int exitCode = process.waitFor();
        if (exitCode != 0) {
            throw new IOException("Python script execution failed with exit code " + exitCode);
        }

        return output.toString();
    }

    private Map<String, Double> loadSemanticSimilarity(String outputDir) {
        File file = new File(appConfig.getAlignOutputPath(), "diagonal_similarity.json");
        if (!file.exists()) {
            return Collections.emptyMap();
        }

        try {
            ObjectMapper objectMapper = new ObjectMapper();
            return objectMapper.readValue(file, new TypeReference<Map<String, Double>>() {});
        } catch (IOException e) {
            throw new RuntimeException("Failed to load semantic similarity", e);
        }
    }
}

