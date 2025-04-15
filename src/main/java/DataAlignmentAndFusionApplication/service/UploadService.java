package DataAlignmentAndFusionApplication.service;

import DataAlignmentAndFusionApplication.model.dto.FileUploadDTO;
import DataAlignmentAndFusionApplication.model.dto.UploadResultDTO;
import DataAlignmentAndFusionApplication.util.Result;


public interface UploadService {
    Result<UploadResultDTO> upload(FileUploadDTO dto);
}
