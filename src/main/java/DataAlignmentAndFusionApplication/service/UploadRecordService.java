package DataAlignmentAndFusionApplication.service;

import DataAlignmentAndFusionApplication.model.dto.FileUploadDTO;
import DataAlignmentAndFusionApplication.model.dto.UploadMessage;
import DataAlignmentAndFusionApplication.model.entity.UploadRecord;
import DataAlignmentAndFusionApplication.model.vo.PageVO;
import DataAlignmentAndFusionApplication.util.Result;
import com.baomidou.mybatisplus.extension.service.IService;

/**
* @author 29857
* @description 针对表【upload_record】的数据库操作Service
* @createDate 2025-04-17 20:05:12
*/
public interface UploadRecordService extends IService<UploadRecord> {
    // 文件上传方法，返回上传结果
    Result<String> uploadFileAndProcess(FileUploadDTO dto);

    //轮询状态和outputPath
    Result<UploadRecord> getTaskStatus(String taskId);

    // 获取已上传列表
    Result<PageVO<UploadRecord>> getUploadList(Integer page, Integer size);

    // 删除记录
    Result<Void> deleteRecord(Long id);

    void createWaitingRecord(UploadMessage message);

    void updateStatus(String taskId, String status);

}
