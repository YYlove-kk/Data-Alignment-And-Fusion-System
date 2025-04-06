package Test.example.batteryScheduling.persistance;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import Test.example.batteryScheduling.domain.HistoryBatteryModule;
import org.springframework.stereotype.Repository;

@Repository
public interface HistoryBatteryModuleMapper extends BaseMapper<HistoryBatteryModule> {
}
