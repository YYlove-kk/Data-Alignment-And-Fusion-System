package Test.example.batteryScheduling.service.impl;

import Test.example.batteryScheduling.DTO.StockDTO;
import Test.example.batteryScheduling.common.CommonResponse;
import Test.example.batteryScheduling.domain.Stock;
import Test.example.batteryScheduling.persistance.StockMapper;
import Test.example.batteryScheduling.service.StockService;
import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import org.springframework.beans.BeanUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service("stockService")
public class StockServiceImpl extends ServiceImpl<StockMapper, Stock> implements StockService {

    @Autowired
    private StockMapper stockMapper;

    @Override
    public CommonResponse<List<Stock>> getAllStock() {
        List<Stock> stocks = stockMapper.selectList(null);
        return CommonResponse.createForSuccess(stocks);
    }

    @Override
    public void updateStock(Stock stock) {
        stockMapper.updateById(stock);
    }

    @Override
    public CommonResponse<Stock> addStock(StockDTO stockDTO) {
        Stock stock = new Stock();
        BeanUtils.copyProperties(stockDTO, stock);
        int result = stockMapper.insert(stock);
        if (result == 1) {
            return CommonResponse.createForSuccess("添加库存成功",stock);
        } else {
            return CommonResponse.createForError("添加库存失败");
        }
    }

    @Override
    public CommonResponse<String> deleteStock(StockDTO stockDTO) {
        QueryWrapper<Stock> queryWrapper = new QueryWrapper<>();
        queryWrapper.eq("set_number", stockDTO.getSetNumber());
        int result = stockMapper.delete(queryWrapper);
        if (result == 1) {
            return CommonResponse.createForSuccess("删除库存成功");
        } else {
            return CommonResponse.createForError("删除库存失败");
        }
    }

    @Override
    public CommonResponse<String> editStock(StockDTO stockDTO) {
        Stock stock = new Stock();
        BeanUtils.copyProperties(stockDTO, stock);
        int result = stockMapper.updateById(stock);
        if (result == 1) {
            return CommonResponse.createForSuccess("库存更新成功");
        } else {
            return CommonResponse.createForError("库存更新失败");
        }
    }
}
