package DataAlignmentAndFusionApplication.service.impl;

import DataAlignmentAndFusionApplication.algorithm.strategy.AnalysisRequest;
import DataAlignmentAndFusionApplication.algorithm.strategy.AnalysisStrategy;
import DataAlignmentAndFusionApplication.algorithm.strategy.AnalysisStrategyFactory;
import DataAlignmentAndFusionApplication.model.dto.AnalysisDTO;
import DataAlignmentAndFusionApplication.model.vo.AnalysisResultVO;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import DataAlignmentAndFusionApplication.model.entity.AnalysisRecord;
import DataAlignmentAndFusionApplication.service.AnalysisRecordService;
import DataAlignmentAndFusionApplication.mapper.module.AnalysisRecordMapper;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

/**
* @author 29857
* @description 针对表【analysis_record】的数据库操作Service实现
* @createDate 2025-04-22 23:49:13
*/
@Service
public class AnalysisRecordServiceImpl extends ServiceImpl<AnalysisRecordMapper, AnalysisRecord>
    implements AnalysisRecordService{
    private final AnalysisRecordMapper recordMapper;
    private final AnalysisStrategyFactory strategyFactory;

    public AnalysisRecordServiceImpl(AnalysisRecordMapper recordMapper, AnalysisStrategyFactory strategyFactory) {
        this.recordMapper = recordMapper;
        this.strategyFactory = strategyFactory;
    }


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
    public Page<AnalysisRecord> getHistory(Long userId, Integer page, Integer size) {
        return null;
    }

    private AnalysisRequest convertToRequest(AnalysisDTO dto) {
        return new AnalysisRequest(        );
    }

    private void saveAnalysisRecord(AnalysisDTO dto, AnalysisResultVO result) {
        AnalysisRecord record = new AnalysisRecord();
//        record.setUserId(dto.getUserId());
//        record.setAnalysisType(dto.getAnalysisType());
//        record.setDataSources(dto.getDataSources());
//        record.setAnalysisMode(dto.getMode());
//        record.setParameters(dto.getCustomParams());
//        record.setResultStats(result.getStats());
        recordMapper.insert(record);
    }

}




