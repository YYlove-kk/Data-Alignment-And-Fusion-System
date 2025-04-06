package Test.example.batteryScheduling.persistance;

import Test.example.batteryScheduling.domain.BatteryCell;
import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import org.springframework.stereotype.Repository;

@Repository
public interface BatteryCellMapper extends BaseMapper<BatteryCell> {
}
