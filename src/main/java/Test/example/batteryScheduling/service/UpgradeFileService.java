package Test.example.batteryScheduling.service;

import Test.example.batteryScheduling.common.CommonResponse;
import com.baomidou.mybatisplus.extension.service.IService;
import Test.example.batteryScheduling.domain.UpgradeFile;
import org.springframework.web.multipart.MultipartFile;

import java.util.Map;

public interface UpgradeFileService extends IService<UpgradeFile> {
    CommonResponse<Boolean> handleFilesUpload(MultipartFile files, String userName);
    CommonResponse<Map<String, String>> getUpgradeByFileNameAndUserName(String applicationName, String upgradePackageName);
}
