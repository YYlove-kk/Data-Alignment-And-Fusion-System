package DataAlignmentAndFusionApplication.service;

import DataAlignmentAndFusionApplication.model.dto.AlignmentRequest;
import DataAlignmentAndFusionApplication.model.entity.AlignmentResult;

public interface AlignmentService {
    AlignmentResult alignTextAndImage() throws Exception;
}