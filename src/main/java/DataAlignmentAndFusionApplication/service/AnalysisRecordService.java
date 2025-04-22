package DataAlignmentAndFusionApplication.service;

import DataAlignmentAndFusionApplication.model.dto.AnalysisDTO;
import DataAlignmentAndFusionApplication.model.entity.AnalysisRecord;
import DataAlignmentAndFusionApplication.model.vo.AnalysisResultVO;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.IService;
import org.springframework.transaction.annotation.Transactional;

/**
* @author 29857
* @description 针对表【analysis_record】的数据库操作Service
* @createDate 2025-04-22 23:49:13
*/
public interface AnalysisRecordService extends IService<AnalysisRecord> {

    @Transactional
    AnalysisResultVO performAnalysis(AnalysisDTO dto);

    Page<AnalysisRecord> getHistory(Long userId, Integer page, Integer size);

}
