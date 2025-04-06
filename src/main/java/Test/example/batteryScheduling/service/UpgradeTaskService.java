package Test.example.batteryScheduling.service;

import Test.example.batteryScheduling.common.CommonResponse;
import Test.example.batteryScheduling.domain.UpgradeTask;
import com.baomidou.mybatisplus.extension.service.IService;
import Test.example.batteryScheduling.DTO.UpgradeTaskAdmitDTO;
import Test.example.batteryScheduling.DTO.UpgradeTaskDTO;
import org.springframework.core.io.Resource;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.RequestBody;

import java.text.ParseException;
import java.util.List;

public interface UpgradeTaskService extends IService<UpgradeTask> {
    CommonResponse<List<UpgradeTaskDTO>> getAllUpgradeTask();
    CommonResponse<Boolean> addUpgradeTask(UpgradeTaskAdmitDTO upgradeTaskAdmitDTO);
    CommonResponse<String> getRandomUpgradeTaskNumber();
    CommonResponse<String> rejectUpgradeTask(UpgradeTaskDTO upgradeTaskDTO,String rejectReason) throws ParseException;
    CommonResponse<String> agreeUpgradeTask(@RequestBody UpgradeTaskDTO upgradeTaskDTO) throws ParseException;

    CommonResponse<List<UpgradeTaskDTO>> getUpgradeTaskHistory();
    CommonResponse<List<UpgradeTaskDTO>> getUpgradeTaskHistoryByUserName(String username);

    ResponseEntity<Resource> getUpgradeFileBMS();
    ResponseEntity<Resource> getUpgradeFileWIL();
    ResponseEntity<Resource> getUpgradeFileVersion();

}
