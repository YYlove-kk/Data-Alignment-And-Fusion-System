package DataAlignmentAndFusionApplication.service.impl;

import DataAlignmentAndFusionApplication.algorithm.strategy.AlgorithmStrategy;
import DataAlignmentAndFusionApplication.mapper.module.AlignmentConfigMapper;
import DataAlignmentAndFusionApplication.mapper.module.AlignmentResultMapper;
import DataAlignmentAndFusionApplication.model.dto.AlignmentConfigDTO;
import DataAlignmentAndFusionApplication.model.dto.AlignmentExecuteDTO;
import DataAlignmentAndFusionApplication.model.entity.AlignmentConfig;
import DataAlignmentAndFusionApplication.model.vo.AlignmentResultVO;
import DataAlignmentAndFusionApplication.model.vo.PageVO;
import DataAlignmentAndFusionApplication.service.AlignmentService;
import DataAlignmentAndFusionApplication.util.Result;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service("alignmentService")
@RequiredArgsConstructor
public class AlignmentServiceImpl implements AlignmentService {
    private final AlignmentConfigMapper configMapper;
    private final AlignmentResultMapper resultMapper;
    private final AlgorithmStrategy strategy;

    @Override
    @Transactional
    public Result<Long> executeAlignment(AlignmentExecuteDTO dto) {
        return null;
    }

    @Override
    public Result<AlignmentResultVO> getResultById(Long resultId) {

        return null;
    }

    @Override
    public Result<AlignmentConfig> getUserConfig(String userId) {
        return null;
    }

    @Override
    public Result<Void> saveConfig(AlignmentConfigDTO dto, Long userId) {
        return null;
    }

    @Override
    public Result<PageVO<AlignmentResultVO>> listHistory(Long userId, Integer page, Integer size) {
        return null;
    }

    // 其他方法...
}
