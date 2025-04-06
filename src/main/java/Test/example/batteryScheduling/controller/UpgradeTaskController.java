package Test.example.batteryScheduling.controller;

import Test.example.batteryScheduling.DTO.UpgradeTaskAdmitDTO;
import Test.example.batteryScheduling.DTO.UpgradeTaskDTO;
import Test.example.batteryScheduling.common.CommonResponse;
import Test.example.batteryScheduling.service.UpgradeTaskService;
import org.springframework.core.io.Resource;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;

import java.text.ParseException;
import java.util.List;

@Validated
@RestController
@RequestMapping("/UpgradeTask")
public class UpgradeTaskController {
    @Autowired
    private UpgradeTaskService upgradeTaskService;

    @RequestMapping("/getAllUpgradeTask")
    public CommonResponse<List<UpgradeTaskDTO>> getAllUpgradeTask(){
        return upgradeTaskService.getAllUpgradeTask();
    }

    @PostMapping("/addUpgradeTask")
    public CommonResponse<Boolean> addUpgradeTask(@RequestBody UpgradeTaskAdmitDTO upgradeTaskAdmitDTO){
        return upgradeTaskService.addUpgradeTask(upgradeTaskAdmitDTO);
    }

    //获取随机申请编号
    @RequestMapping("/getRandomUpgradeTaskNumber")
    public CommonResponse<String> getRandomUpgradeTaskNumber(){
        return upgradeTaskService.getRandomUpgradeTaskNumber();
    }

    //拒绝升级任务请求
    @RequestMapping("/rejectUpgradeTask")
    public CommonResponse<String> rejectUpgradeTask(@RequestBody UpgradeTaskDTO upgradeTaskDTO,@RequestParam String rejectReason) throws ParseException {
        return upgradeTaskService.rejectUpgradeTask(upgradeTaskDTO,rejectReason);
    }

    //拒绝升级任务请求
    @RequestMapping("/agreeUpgradeTask")
    public CommonResponse<String> agreeUpgradeTask(@RequestBody UpgradeTaskDTO upgradeTaskDTO) throws ParseException {
        return upgradeTaskService.agreeUpgradeTask(upgradeTaskDTO);
    }

    @RequestMapping("/getUpgradeTaskHistory")
    public CommonResponse<List<UpgradeTaskDTO>> getUpgradeTaskHistory(){
        return upgradeTaskService.getUpgradeTaskHistory();
    }

    @RequestMapping("/getUpgradeTaskHistoryByUserName")
    public CommonResponse<List<UpgradeTaskDTO>> getUpgradeTaskHistoryByUserName(@RequestParam String username){
        return upgradeTaskService.getUpgradeTaskHistoryByUserName(username);
    }

    @GetMapping("/getUpgradeFileBMS")
    public ResponseEntity<Resource> getUpgradeFileBMS() {
        return upgradeTaskService.getUpgradeFileBMS();
    }
    @GetMapping("/getUpgradeFileWIL")
    public ResponseEntity<Resource> getUpgradeFileWIL() {
        return upgradeTaskService.getUpgradeFileWIL();
    }
    @GetMapping("/getUpgradeFileVersion")
    public ResponseEntity<Resource> getUpgradeFileVersion() {
        return upgradeTaskService.getUpgradeFileVersion();
    }
}
