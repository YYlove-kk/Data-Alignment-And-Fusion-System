package DataAlignmentAndFusionApplication.util;

import DataAlignmentAndFusionApplication.model.entity.JointEmbeddingRelation;
import DataAlignmentAndFusionApplication.service.JointEmbeddingRelationService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

@Component
public class CsvImporter {

    @Autowired
    private JointEmbeddingRelationService relationService;

    public void importFromCsv(String csvPath) throws IOException {
        List<JointEmbeddingRelation> relations = new ArrayList<>();

        try (BufferedReader reader = new BufferedReader(new FileReader(csvPath))) {
            String line = reader.readLine(); // skip header
            while ((line = reader.readLine()) != null) {
                String[] parts = line.split(",");
                if (parts.length >= 3) {
                    String jointId = parts[0]; // e.g., joint_patientA_00001.npy
                    String patientId = extractPatientId(jointId);
                    String textFile = parts[1];
                    String imageFile = parts[2];

                    JointEmbeddingRelation relation = new JointEmbeddingRelation();
                    relation.setPatientId(patientId);
                    relation.setTextFile(textFile);
                    relation.setImageFile(imageFile);
                    relations.add(relation);
                }
            }
        }

        relationService.saveBatch(relations);
        System.out.println("✅ 数据批量导入完成，记录数: " + relations.size());
    }

    private String extractPatientId(String jointId) {
        Pattern pattern = Pattern.compile("joint_(.*?)_\\d+\\.npy");
        Matcher matcher = pattern.matcher(jointId);
        if (matcher.matches()) {
            return matcher.group(1);
        }
        return "UNKNOWN";
    }
}
