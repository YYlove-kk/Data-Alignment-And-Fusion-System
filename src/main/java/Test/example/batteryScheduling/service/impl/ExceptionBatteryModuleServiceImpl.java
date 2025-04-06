package Test.example.batteryScheduling.service.impl;

import Test.example.batteryScheduling.persistance.ExceptionBatteryModuleMapper;
import Test.example.batteryScheduling.service.ExceptionBatteryModuleService;
import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import Test.example.batteryScheduling.DTO.ExceptionBatteryModuleDTO;
import Test.example.batteryScheduling.common.CommonResponse;
import Test.example.batteryScheduling.domain.ExceptionBatteryModule;
import org.springframework.beans.BeanUtils;
import org.springframework.stereotype.Service;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

@Validated
@CrossOrigin
@RestController
@RequestMapping("/ExceptionBatteryModule")
@Service("exceptionBatteryModuleService")
public class ExceptionBatteryModuleServiceImpl extends ServiceImpl<ExceptionBatteryModuleMapper, ExceptionBatteryModule> implements ExceptionBatteryModuleService {


    @Override
    public CommonResponse<List<ExceptionBatteryModuleDTO>> getAllExceptionBatteryModule() {
        QueryWrapper<ExceptionBatteryModule> wrapper = new QueryWrapper<>();
        System.out.println("Search All ExceptionModule");
        List<ExceptionBatteryModule> exceptionBatteryModuleList = list(wrapper);
        if (exceptionBatteryModuleList != null && !exceptionBatteryModuleList.isEmpty()) {
            List<ExceptionBatteryModuleDTO> exceptionBatteryModuleDTOList = exceptionBatteryModuleList.stream().map(exceptionBatteryModule -> {
                ExceptionBatteryModuleDTO dto = new ExceptionBatteryModuleDTO();
                BeanUtils.copyProperties(exceptionBatteryModule, dto);
                // 特别处理id，将其转换为String
                dto.setId(String.valueOf(exceptionBatteryModule.getId()));
                return dto;
            }).collect(Collectors.toList());

            System.out.println("恒温房异常设备数为：" + exceptionBatteryModuleDTOList.size());
            // 仅用于日志，显示第一个DTO的ID，确认其为String类型
            System.out.println("FID: " + exceptionBatteryModuleDTOList.get(0).getId());
            return CommonResponse.createForSuccess("success", exceptionBatteryModuleDTOList);
        } else {
            return CommonResponse.createForError("恒温房异常设备数为0");
        }
    }
    @Override
    public CommonResponse<Void> batchAddExceptionBatteryModules(List<ExceptionBatteryModule> exceptionBatteryModules) {
        String[] wilVersions = {"V1.1.4", "V2.0.0", "V2.0.4", "V2.0.8", "V2.0.9"};
        String[] bmsScriptVersions = {"V10", "V11", "V12"};
        String[] partNumbers = {"24120503", "24120504", "24120149", "24120150", "24120443", "24120444"};
        String[] moduleNames = {"NCM 2P8S 模组总成 A", "NCM 2P8S 模组总成 B", "NCM 1P12S 模组总成A", "NCM 1P12S 模组总成 B", "LFP 1P12S 模组总成A", "LFP 1P12S 模组总成 B"};
        Random random = new Random();

        List<ExceptionBatteryModule> exceptionBatteryModuleList = new ArrayList<>();

        for (int i = 0; i < 30; i++) {
            ExceptionBatteryModule batteryModule = new ExceptionBatteryModule();
            int partIndex = random.nextInt(partNumbers.length);
            batteryModule.setPartNumber(partNumbers[partIndex]);
            batteryModule.setModuleName(moduleNames[partIndex]);
            batteryModule.setSetNumber("BEV(1416)S" + (10000 + random.nextInt(90000)));
            batteryModule.setMacAddress("MAC-" + Long.toHexString(Double.doubleToLongBits(random.nextDouble())));
            batteryModule.setModuleVersion(wilVersions[random.nextInt(wilVersions.length)]);
            batteryModule.setScriptVersion(bmsScriptVersions[random.nextInt(bmsScriptVersions.length)]);
            batteryModule.setTime(new Date()); // Current date and time

            // Randomly determine the type of exception
            int exceptionType = random.nextInt(5) + 1;
            batteryModule.setExceptionEntryStatus(String.valueOf(exceptionType));

            switch (exceptionType) {
                case 1: // SOC异常
                    batteryModule.setModuleSoc(random.nextInt(21)); // SOC [0, 20]
                    batteryModule.setModuleTemperature(random.nextInt(29) + 1); // 温度 [1, 29]
                    break;
                case 2: // 温度异常
                    batteryModule.setModuleSoc(random.nextInt(80) + 21); // SOC [21, 100]
                    // 温度 [-10, 0] or [30, 40]
                    batteryModule.setModuleTemperature(random.nextBoolean() ? random.nextInt(11) - 10 : random.nextInt(11) + 30);
                    break;
                case 3: // 自放电异常
                case 4: // 通讯异常
                case 5: // 性能异常
                    batteryModule.setModuleSoc(random.nextInt(80) + 21); // SOC [21, 100]
                    batteryModule.setModuleTemperature(random.nextInt(29) + 1); // 温度 [1, 29]
                    break;
                default:
                    // Should not reach here
                    break;
            }

            exceptionBatteryModuleList.add(batteryModule);
        }

        // Assuming saveBatch(batteryModules) is the method you would call to save these generated modules
        // This line is just a placeholder for wherever you would actually persist these modules
        boolean saved = saveBatch(exceptionBatteryModuleList);
        return saved ? CommonResponse.createForSuccessMessage("批量添加异常状态模组成功")
                : CommonResponse.createForError("批量添加异常状态模组失败");
    }



}
