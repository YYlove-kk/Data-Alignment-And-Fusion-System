package DataAlignmentAndFusionApplication.service.impl;

import DataAlignmentAndFusionApplication.algorithm.strategy.AnalysisRequest;
import DataAlignmentAndFusionApplication.algorithm.strategy.AnalysisStrategy;
import DataAlignmentAndFusionApplication.algorithm.strategy.AnalysisStrategyFactory;
import DataAlignmentAndFusionApplication.mapper.module.PatientAnalysisRecordMapper;
import DataAlignmentAndFusionApplication.model.dto.AnalysisDTO;
import DataAlignmentAndFusionApplication.model.entity.PatientAnalysisRecord;
import DataAlignmentAndFusionApplication.model.vo.AnalysisResultVO;
import DataAlignmentAndFusionApplication.service.AnalysisService;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
@RequiredArgsConstructor
public class AnalysisServiceImpl implements AnalysisService {
    private final PatientAnalysisRecordMapper recordMapper;
    private final AnalysisStrategyFactory strategyFactory;

    @Transactional
    @Override
    public AnalysisResultVO performAnalysis(AnalysisDTO dto) {
        // 1. 获取分析策略
        AnalysisStrategy strategy = strategyFactory.getStrategy(dto.getAnalysisType());

        // 2. 执行分析
        AnalysisRequest request = convertToRequest(dto);
        AnalysisResultVO result = strategy.analyze(request);

        // 3. 保存记录
        saveAnalysisRecord(dto, result);

        return result;
    }

    @Override
    public Page<PatientAnalysisRecord> getHistory(Long userId, Integer page, Integer size) {
        return null;
    }

    private AnalysisRequest convertToRequest(AnalysisDTO dto) {
        return new AnalysisRequest(        );
    }

    private void saveAnalysisRecord(AnalysisDTO dto, AnalysisResultVO result) {
        PatientAnalysisRecord record = new PatientAnalysisRecord();
//        record.setUserId(dto.getUserId());
//        record.setAnalysisType(dto.getAnalysisType());
//        record.setDataSources(dto.getDataSources());
//        record.setAnalysisMode(dto.getMode());
//        record.setParameters(dto.getCustomParams());
//        record.setResultStats(result.getStats());
        recordMapper.insert(record);
    }
}
