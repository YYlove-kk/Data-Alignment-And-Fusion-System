package Test.example.batteryScheduling.persistance;

import Test.example.batteryScheduling.domain.Region;
import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import org.springframework.stereotype.Repository;

@Repository
public interface RegionMapper extends BaseMapper<Region> {
}
