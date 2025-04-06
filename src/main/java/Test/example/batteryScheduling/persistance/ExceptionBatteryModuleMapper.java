package Test.example.batteryScheduling.persistance;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import Test.example.batteryScheduling.domain.ExceptionBatteryModule;
import org.springframework.stereotype.Repository;

@Repository
public interface ExceptionBatteryModuleMapper extends BaseMapper<ExceptionBatteryModule> {
}
