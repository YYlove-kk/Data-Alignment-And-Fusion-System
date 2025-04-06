package Test.example.batteryScheduling.persistance;

import Test.example.batteryScheduling.domain.Device;
import Test.example.batteryScheduling.domain.Stock;
import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import org.springframework.stereotype.Repository;

@Repository
public interface StockMapper extends BaseMapper<Stock> {
}
