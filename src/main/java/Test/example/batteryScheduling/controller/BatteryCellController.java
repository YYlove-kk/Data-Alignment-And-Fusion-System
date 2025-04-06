package Test.example.batteryScheduling.controller;

import Test.example.batteryScheduling.DTO.BatteryCellDTO;
import Test.example.batteryScheduling.DTO.BatteryModuleDTO;
import Test.example.batteryScheduling.DTO.HistoryBatteryModuleDTO;
import Test.example.batteryScheduling.common.CommonResponse;
import Test.example.batteryScheduling.domain.BatteryCell;
import Test.example.batteryScheduling.domain.HistoryBatteryModule;
import Test.example.batteryScheduling.service.BatteryCellService;
import Test.example.batteryScheduling.service.HistoryBatteryModuleService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@Validated
@RestController
@RequestMapping("/BatteryCell")
public class BatteryCellController {
    @Autowired
    BatteryCellService batteryCellService;
    @RequestMapping("/getInfosByMac")
    public CommonResponse<List<BatteryCellDTO>> getInfosByMac(@RequestParam String macAddress){
        return batteryCellService.getInfosByMac(macAddress);
    }
}