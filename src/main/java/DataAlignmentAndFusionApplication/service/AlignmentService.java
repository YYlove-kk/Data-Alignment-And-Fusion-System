package DataAlignmentAndFusionApplication.service;

import DataAlignmentAndFusionApplication.model.dto.AlignmentConfigDTO;
import DataAlignmentAndFusionApplication.model.dto.AlignmentExecuteDTO;
import DataAlignmentAndFusionApplication.model.entity.AlignmentConfig;
import DataAlignmentAndFusionApplication.model.vo.AlignmentResultVO;
import DataAlignmentAndFusionApplication.model.vo.PageVO;
import DataAlignmentAndFusionApplication.util.Result;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface AlignmentService {
    /**
     * 执行数据对齐
     */
    Result<Long> executeAlignment(AlignmentExecuteDTO dto);


    /**
     * 获取对齐结果
     * @param resultId
     * @return
     */
    Result<AlignmentResultVO> getResultById(Long resultId);

    /**
     * 获取用户历史配置
     * @param userId
     * @return
     */
    Result<AlignmentConfig> getUserConfig(String userId);

    Result<Void> saveConfig(AlignmentConfigDTO dto, Long userId);

    /**
     * 分页查询历史记录
     */
    Result<PageVO<AlignmentResultVO>> listHistory(Long userId, Integer page, Integer size);
}
