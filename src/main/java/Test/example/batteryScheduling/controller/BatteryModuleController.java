package Test.example.batteryScheduling.controller;

import Test.example.batteryScheduling.DTO.BatteryModuleDTO;
import Test.example.batteryScheduling.DTO.ExceptionBatteryModuleDTO;
import Test.example.batteryScheduling.common.CommonResponse;
import Test.example.batteryScheduling.domain.BatteryModule;
import Test.example.batteryScheduling.domain.ExceptionBatteryModule;
import Test.example.batteryScheduling.service.BatteryModuleService;
import Test.example.batteryScheduling.service.ExceptionBatteryModuleService;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

@Validated
@RestController
@RequestMapping("/BatteryModule")
public class BatteryModuleController {
    @Autowired
    private BatteryModuleService batteryModuleService;
    @Autowired
    private ExceptionBatteryModuleService exceptionBatteryModuleService;

    @RequestMapping("/infos")
    public CommonResponse<List<BatteryModuleDTO>> getAllBatteryModule(){
        return batteryModuleService.getAllBatteryModule();
    }

    @RequestMapping("/paginatedInfos")
    public CommonResponse<Map<String, Object>> getPaginatedBatteryModules(
            @RequestParam(defaultValue = "1") int page,
            @RequestParam(defaultValue = "10") int size) {
        return batteryModuleService.getPaginatedBatteryModules(page, size);
    }


    @RequestMapping("/exceptionInfos")
    public CommonResponse<List<ExceptionBatteryModuleDTO>> getAllExceptionBatteryModule(){
        return exceptionBatteryModuleService.getAllExceptionBatteryModule();
    }

    //更新模组状态
    @RequestMapping("/updateStatus")
    public CommonResponse<Void> updateBatteryModuleStatus(){
        //TODO 增加判断逻辑，只查询异常的设备
        return batteryModuleService.updateBatteryModuleStatus();
    }

    @PostMapping("/batchAdd")
    public CommonResponse<Void> batchAddBatteryModules(@RequestBody List<BatteryModuleDTO> batteryModuleDTOs) {
        return batteryModuleService.batchAddBatteryModules(batteryModuleDTOs);
    }


    @PostMapping("/batchAddException")
    public CommonResponse<Void> batchAddExceptionBatteryModules() {
        System.out.println("BATCH ADD Exception");
        List<ExceptionBatteryModule> exceptionBatteryModules = new ArrayList<ExceptionBatteryModule>();
        return exceptionBatteryModuleService.batchAddExceptionBatteryModules(exceptionBatteryModules);
    }

    @PostMapping("/delete")
    public CommonResponse<Void> deleteBatteryModules(@RequestBody String ids) throws JsonProcessingException {
        System.out.println("IDS:" + ids);
        ObjectMapper mapper = new ObjectMapper();
        List<String> idList = mapper.readValue(ids, new TypeReference<List<String>>(){});
        System.out.println("List:" + idList.get(0));
        return batteryModuleService.batchSoftDeleteBatteryModules(idList);
    }
}
