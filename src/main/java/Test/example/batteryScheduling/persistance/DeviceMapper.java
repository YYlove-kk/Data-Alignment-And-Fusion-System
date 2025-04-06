package Test.example.batteryScheduling.persistance;

import Test.example.batteryScheduling.domain.Device;
import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import org.springframework.stereotype.Repository;

@Repository
public interface DeviceMapper extends BaseMapper<Device> {
}
