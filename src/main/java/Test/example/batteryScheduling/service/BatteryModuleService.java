package Test.example.batteryScheduling.service;

import Test.example.batteryScheduling.DTO.BatteryModuleDTO;
import Test.example.batteryScheduling.common.CommonResponse;
import Test.example.batteryScheduling.domain.BatteryModule;
import com.baomidou.mybatisplus.extension.service.IService;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.Map;

public interface BatteryModuleService extends IService<BatteryModule> {
    CommonResponse<List<BatteryModuleDTO>> getAllBatteryModule();
    CommonResponse<Void> batchAddBatteryModules(List<BatteryModuleDTO> batteryModules);

    CommonResponse<Void> batchSoftDeleteBatteryModules(List<String> ids);
    CommonResponse<Void> updateBatteryModuleStatus();

    BatteryModule selectBatteryModuleByMac(String macAddress);

    CommonResponse<Map<String, Object>> getPaginatedBatteryModules(int page, int size);

}
