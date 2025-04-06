package Test.example.batteryScheduling.persistance;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import Test.example.batteryScheduling.domain.BatteryModule;
import org.springframework.stereotype.Repository;

@Repository
public interface BatteryModuleMapper extends BaseMapper<BatteryModule> {
}
