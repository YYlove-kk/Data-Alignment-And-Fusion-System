package Test.example.batteryScheduling.persistance;

import Test.example.batteryScheduling.domain.UpgradeTask;
import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import org.springframework.stereotype.Repository;

@Repository
public interface UpgradeTaskMapper extends BaseMapper<UpgradeTask> {
}
