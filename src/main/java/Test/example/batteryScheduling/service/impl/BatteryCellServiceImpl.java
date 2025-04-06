package Test.example.batteryScheduling.service.impl;

import Test.example.batteryScheduling.DTO.BatteryCellDTO;
import Test.example.batteryScheduling.DTO.BatteryModuleDTO;
import Test.example.batteryScheduling.common.CommonResponse;
import Test.example.batteryScheduling.domain.BatteryCell;
import Test.example.batteryScheduling.domain.BatteryModule;
import Test.example.batteryScheduling.domain.HistoryBatteryModule;
import Test.example.batteryScheduling.persistance.BatteryCellMapper;
import Test.example.batteryScheduling.persistance.BatteryModuleMapper;
import Test.example.batteryScheduling.service.BatteryCellService;
import Test.example.batteryScheduling.service.BatteryModuleService;
import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import org.springframework.beans.BeanUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;
import java.util.stream.Collectors;

@Validated
@CrossOrigin
@RestController
@RequestMapping("/BatteryCell")
@Service("batteryCellService")
public class BatteryCellServiceImpl extends ServiceImpl<BatteryCellMapper, BatteryCell> implements BatteryCellService {
    @Override
    public CommonResponse<List<BatteryCellDTO>> getInfosByMac(String macAddress) {
        System.out.println("BTC:" + macAddress);
        QueryWrapper<BatteryCell> queryWrapper = new QueryWrapper<>();
        queryWrapper.eq("mac_address", macAddress);
        List<BatteryCell> cells = this.list(queryWrapper);

        // 将 domain 转换为 DTO
        List<BatteryCellDTO> cellDTOs = cells.stream().map(cell -> {
            BatteryCellDTO dto = new BatteryCellDTO();
            BeanUtils.copyProperties(cell, dto);
            return dto;
        }).collect(Collectors.toList());

        return CommonResponse.createForSuccess("查询成功", cellDTOs);
    }
}
