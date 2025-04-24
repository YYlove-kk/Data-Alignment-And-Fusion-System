package DataAlignmentAndFusionApplication.service;

import DataAlignmentAndFusionApplication.common.UploadResp;
import DataAlignmentAndFusionApplication.model.dto.FileUploadDTO;
import DataAlignmentAndFusionApplication.model.dto.UploadResultDTO;
import DataAlignmentAndFusionApplication.model.entity.UploadRecord;
import DataAlignmentAndFusionApplication.model.vo.UploadRecordVO;
import DataAlignmentAndFusionApplication.util.Result;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.IService;

/**
* @author 29857
* @description 针对表【upload_record】的数据库操作Service
* @createDate 2025-04-17 20:05:12
*/
public interface UploadRecordService extends IService<UploadRecord> {
    // 文件上传方法，返回上传结果
    UploadResp uploadFileAndProcess(FileUploadDTO dto);

    // 获取已上传列表
    Result<Page<UploadRecordVO>> getUploadList(Integer page, Integer size);

    // 删除记录
    Result<Void> deleteRecord(Long id);
}
