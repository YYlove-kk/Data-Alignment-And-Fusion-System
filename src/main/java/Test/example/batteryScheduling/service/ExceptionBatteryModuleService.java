package Test.example.batteryScheduling.service;

import com.baomidou.mybatisplus.extension.service.IService;
import Test.example.batteryScheduling.DTO.ExceptionBatteryModuleDTO;
import Test.example.batteryScheduling.common.CommonResponse;
import Test.example.batteryScheduling.domain.ExceptionBatteryModule;

import java.util.List;

public interface ExceptionBatteryModuleService extends IService<ExceptionBatteryModule> {
    CommonResponse<List<ExceptionBatteryModuleDTO>> getAllExceptionBatteryModule();
    CommonResponse<Void> batchAddExceptionBatteryModules(List<ExceptionBatteryModule> exceptionBatteryModules);
}
