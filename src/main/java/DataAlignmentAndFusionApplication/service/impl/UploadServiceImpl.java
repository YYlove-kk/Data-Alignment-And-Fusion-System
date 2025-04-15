package DataAlignmentAndFusionApplication.service.impl;

import DataAlignmentAndFusionApplication.mapper.module.UploadRecordMapper;
import DataAlignmentAndFusionApplication.model.dto.FileUploadDTO;
import DataAlignmentAndFusionApplication.model.dto.UploadResultDTO;
import DataAlignmentAndFusionApplication.model.entity.UploadRecord;
import DataAlignmentAndFusionApplication.service.UploadService;
import DataAlignmentAndFusionApplication.util.Result;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;

@Service("uploadService")
public class UploadServiceImpl implements UploadService {
    @Autowired
    private UploadRecordMapper recordMapper;

    @Override
    public Result<UploadResultDTO> upload(FileUploadDTO dto) {
        return null;
    }

    @Async
    public void asyncProcessFile(UploadRecord record) {

    }
}
