package DataAlignmentAndFusionApplication.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

import java.time.LocalDateTime;
import java.util.Map;
import java.util.Set;

@Data
@TableName("patient_analysis_record")
public class PatientAnalysisRecord {
    @TableId(type = IdType.ASSIGN_ID)
    private Long id;

    @NotNull
    private Long userId;          // 操作用户ID

    @NotBlank
    private String analysisType;  // single_patient/disease

    private Set<String> dataSources; // 数据来源集合（如["hospital_A", "clinic_B"]）

    @NotBlank
    private String analysisMode;  // basic/advanced

    private Map<String, Object> parameters; // 分析参数

    private Map<String, Object> resultStats; // 统计结果

    private LocalDateTime createTime;
}
