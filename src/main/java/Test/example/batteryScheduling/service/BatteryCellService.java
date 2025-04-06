package Test.example.batteryScheduling.service;

import Test.example.batteryScheduling.DTO.BatteryCellDTO;
import Test.example.batteryScheduling.DTO.BatteryModuleDTO;
import Test.example.batteryScheduling.common.CommonResponse;
import Test.example.batteryScheduling.domain.BatteryCell;
import Test.example.batteryScheduling.domain.BatteryModule;
import Test.example.batteryScheduling.domain.HistoryBatteryModule;
import com.baomidou.mybatisplus.extension.service.IService;
import org.springframework.web.bind.annotation.RequestParam;

import java.util.List;

public interface BatteryCellService extends IService<BatteryCell> {
    CommonResponse<List<BatteryCellDTO>> getInfosByMac(@RequestParam String macAddress);
}
