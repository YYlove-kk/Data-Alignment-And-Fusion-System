package DataAlignmentAndFusionApplication.service;

import DataAlignmentAndFusionApplication.model.DTO.FileUploadDTO;
import DataAlignmentAndFusionApplication.model.DTO.UploadResultDTO;
import DataAlignmentAndFusionApplication.util.Result;


public interface UploadService {
    Result<UploadResultDTO> upload(FileUploadDTO dto);
}
