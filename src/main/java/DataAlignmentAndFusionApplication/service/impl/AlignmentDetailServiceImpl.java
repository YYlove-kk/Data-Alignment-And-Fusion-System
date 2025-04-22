package DataAlignmentAndFusionApplication.service.impl;

import DataAlignmentAndFusionApplication.algorithm.strategy.AlgorithmStrategy;
import DataAlignmentAndFusionApplication.mapper.module.AlignmentConfigMapper;
import DataAlignmentAndFusionApplication.mapper.module.AlignmentResultMapper;
import DataAlignmentAndFusionApplication.model.dto.AlignmentConfigDTO;
import DataAlignmentAndFusionApplication.model.dto.AlignmentExecuteDTO;
import DataAlignmentAndFusionApplication.model.entity.AlignmentConfig;
import DataAlignmentAndFusionApplication.model.vo.AlignmentResultVO;
import DataAlignmentAndFusionApplication.model.vo.PageVO;
import DataAlignmentAndFusionApplication.util.Result;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import DataAlignmentAndFusionApplication.model.entity.AlignmentDetail;
import DataAlignmentAndFusionApplication.service.AlignmentDetailService;
import DataAlignmentAndFusionApplication.mapper.module.AlignmentDetailMapper;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

/**
* @author 29857
* @description 针对表【alignment_detail】的数据库操作Service实现
* @createDate 2025-04-22 23:39:40
*/
@Service
public class AlignmentDetailServiceImpl extends ServiceImpl<AlignmentDetailMapper, AlignmentDetail>
    implements AlignmentDetailService{

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


}




