package Test.example.batteryScheduling.controller;

import Test.example.batteryScheduling.DTO.HistoryBatteryModuleDTO;
import Test.example.batteryScheduling.common.CommonResponse;
import Test.example.batteryScheduling.domain.HistoryBatteryModule;
import Test.example.batteryScheduling.service.HistoryBatteryModuleService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@Validated
@RestController
@RequestMapping("/HistoryBatteryModule")
public class HistoryBatteryModuleController {
    @Autowired
    HistoryBatteryModuleService historyBatteryModuleService;
    @RequestMapping("/infos")
    public CommonResponse<List<HistoryBatteryModule>> getHistoryBatteryModule(@RequestBody HistoryBatteryModuleDTO historyBatteryModuleDTO){
        return historyBatteryModuleService.getHistoryBatteryModule(historyBatteryModuleDTO.getMacAddress(),historyBatteryModuleDTO.getStartTime(),historyBatteryModuleDTO.getEndTime());
    }
}
