package DataAlignmentAndFusionApplication.service;

import DataAlignmentAndFusionApplication.model.dto.AnalysisDTO;
import DataAlignmentAndFusionApplication.model.entity.PatientAnalysisRecord;
import DataAlignmentAndFusionApplication.model.vo.AnalysisResultVO;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import org.springframework.transaction.annotation.Transactional;

public interface AnalysisService {
    @Transactional
    AnalysisResultVO performAnalysis(AnalysisDTO dto);

    Page<PatientAnalysisRecord> getHistory(Long userId, Integer page, Integer size);
}
