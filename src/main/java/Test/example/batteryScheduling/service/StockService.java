package Test.example.batteryScheduling.service;

import Test.example.batteryScheduling.DTO.DeviceDTO;
import Test.example.batteryScheduling.DTO.StockDTO;
import Test.example.batteryScheduling.common.CommonResponse;
import Test.example.batteryScheduling.domain.Device;
import Test.example.batteryScheduling.domain.Stock;
import com.baomidou.mybatisplus.extension.service.IService;

import java.util.List;

public interface StockService extends IService<Stock> {
    CommonResponse<List<Stock>> getAllStock();

    void updateStock(Stock stock);

    CommonResponse<Stock> addStock(StockDTO stockDTO);
    CommonResponse<String> deleteStock(StockDTO stockDTO);

    CommonResponse<String> editStock(StockDTO stockDTO);
}
