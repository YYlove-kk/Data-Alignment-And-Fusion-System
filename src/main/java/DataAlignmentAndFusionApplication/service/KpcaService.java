package DataAlignmentAndFusionApplication.service;

import DataAlignmentAndFusionApplication.model.dto.GraphReq;
import DataAlignmentAndFusionApplication.util.Result;

public interface KpcaService {
    /**
     * 执行 KPCA 降维并返回结果
     */
    Result<String> runKpcaReduction(GraphReq req);
}
