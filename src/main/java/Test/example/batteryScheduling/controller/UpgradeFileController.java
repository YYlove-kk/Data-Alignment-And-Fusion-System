package Test.example.batteryScheduling.controller;

import Test.example.batteryScheduling.common.CommonResponse;
import Test.example.batteryScheduling.service.UpgradeFileService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.Map;

@Validated
@RestController
@RequestMapping("/UpgradeFile")
public class UpgradeFileController {
    @Autowired
    private UpgradeFileService upgradeFileService;

    @PostMapping
    @RequestMapping("/uploadFiles")
    public CommonResponse<Boolean> uploadFiles(@RequestParam("file") MultipartFile file, @RequestParam("userName") String userName) {
        return upgradeFileService.handleFilesUpload(file, userName);
    }

    //管理员下载文件
    @GetMapping("/getUpgradeByFileNameAndUserName")
    public CommonResponse<Map<String, String>> getUpgradeByFileNameAndUserName(@RequestParam String applicationName, @RequestParam String upgradePackageName) {
        return upgradeFileService.getUpgradeByFileNameAndUserName(applicationName, upgradePackageName);
    }

}
