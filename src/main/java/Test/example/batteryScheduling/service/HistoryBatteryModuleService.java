package Test.example.batteryScheduling.service;

import Test.example.batteryScheduling.common.CommonResponse;
import com.baomidou.mybatisplus.extension.service.IService;
import Test.example.batteryScheduling.DTO.HistoryBatteryModuleDTO;
import Test.example.batteryScheduling.domain.HistoryBatteryModule;

import java.util.Date;
import java.util.List;

public interface HistoryBatteryModuleService extends IService<HistoryBatteryModule> {
    CommonResponse<List<HistoryBatteryModule>> getHistoryBatteryModule(String macAddress, Date startTime, Date endTime);
}
