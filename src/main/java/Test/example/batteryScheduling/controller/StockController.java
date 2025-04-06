package Test.example.batteryScheduling.controller;

import Test.example.batteryScheduling.DTO.DeviceDTO;
import Test.example.batteryScheduling.DTO.StockDTO;
import Test.example.batteryScheduling.common.CommonResponse;
import Test.example.batteryScheduling.domain.Device;
import Test.example.batteryScheduling.domain.Stock;
import Test.example.batteryScheduling.service.DeviceService;
import Test.example.batteryScheduling.service.StockService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@Validated
@RestController
@RequestMapping("/Stock")
public class StockController {
    @Autowired
    private StockService stockService;

    @RequestMapping("/infos")
    public CommonResponse<List<Stock>> getAllStock(){
        return stockService.getAllStock();
    }

    @PostMapping("/addStock")
    public CommonResponse<Stock> addDevice(@RequestBody StockDTO stockDTO) {
        // 这里调用 service 层处理注册逻辑
        return stockService.addStock(stockDTO);
    }

    @PostMapping("/editStock")
    public CommonResponse<String> editDevice(@RequestBody StockDTO stockDTO) {
        // 这里调用 service 层处理注册逻辑
        return stockService.editStock(stockDTO);
    }

    @PostMapping("/deleteStock")
    public CommonResponse<String> deleteDevice(@RequestBody StockDTO stockDTO) {
        // 这里调用 service 层处理注册逻辑
        return stockService.deleteStock(stockDTO);
    }

}
