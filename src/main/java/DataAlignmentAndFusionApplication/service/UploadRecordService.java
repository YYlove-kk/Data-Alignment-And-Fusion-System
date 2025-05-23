package DataAlignmentAndFusionApplication.service;

import DataAlignmentAndFusionApplication.model.dto.UploadMessage;
import DataAlignmentAndFusionApplication.model.dto.UploadReq;
import DataAlignmentAndFusionApplication.model.entity.UploadRecord;
import DataAlignmentAndFusionApplication.model.vo.PageVO;
import DataAlignmentAndFusionApplication.util.Result;
import com.baomidou.mybatisplus.extension.service.IService;

import java.util.List;

/**
* @author 29857
* @description 针对表【upload_record】的数据库操作Service
* @createDate 2025-04-17 20:05:12
*/
public interface UploadRecordService extends IService<UploadRecord> {
    // 文件上传方法，返回上传结果
    Result<String> uploadFile(UploadReq req);

    Result<String> embedFile();

    // 获取已上传列表
    Result<PageVO<UploadRecord>> getUploadList(Integer page, Integer size);

    // 删除记录
    Result<Void> deleteRecord(Long id);

    void createWaitingRecord(UploadMessage message);

    void updateStatus(String taskId, String status);

    void updatePaths(String taskId, String cleanPath, List<String> npyPaths);

}
