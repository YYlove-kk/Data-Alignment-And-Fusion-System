package DataAlignmentAndFusionApplication.service;

import DataAlignmentAndFusionApplication.model.vo.AlignmentVO;
import DataAlignmentAndFusionApplication.util.Result;

import java.util.List;

public interface AlignmentService {
    Result<String> alignTextAndImage() throws Exception;

    List<AlignmentVO> getAllResults();
}