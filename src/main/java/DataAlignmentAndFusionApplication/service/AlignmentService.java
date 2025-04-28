package DataAlignmentAndFusionApplication.service;

import DataAlignmentAndFusionApplication.model.dto.AlignmentRequest;
import DataAlignmentAndFusionApplication.model.entity.AlignmentRecord;
import DataAlignmentAndFusionApplication.model.entity.AlignmentResult;

import java.util.List;

public interface AlignmentService {
    AlignmentResult alignTextAndImage() throws Exception;

    List<AlignmentRecord> getAllResults();
}