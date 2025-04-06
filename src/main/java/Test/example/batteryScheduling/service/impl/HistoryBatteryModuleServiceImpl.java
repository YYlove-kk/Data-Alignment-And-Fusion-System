package Test.example.batteryScheduling.service.impl;

import Test.example.batteryScheduling.common.CommonResponse;
import Test.example.batteryScheduling.service.HistoryBatteryModuleService;
import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import Test.example.batteryScheduling.DTO.HistoryBatteryModuleDTO;
import Test.example.batteryScheduling.domain.BatteryModule;
import Test.example.batteryScheduling.domain.HistoryBatteryModule;
import Test.example.batteryScheduling.persistance.HistoryBatteryModuleMapper;
import Test.example.batteryScheduling.service.BatteryModuleService;
import org.junit.Test;
import org.springframework.beans.BeanUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.Calendar;
import java.util.Date;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

@Validated
@CrossOrigin
@RestController
@RequestMapping("/HistoryBatteryModule")
@Service("historyBatteryModuleService")
public class HistoryBatteryModuleServiceImpl extends ServiceImpl<HistoryBatteryModuleMapper, HistoryBatteryModule> implements HistoryBatteryModuleService {

    @Autowired
    private HistoryBatteryModuleMapper historyBatteryModuleMapper;
    @Override
    public CommonResponse<List<HistoryBatteryModule>> getHistoryBatteryModule(String macAddress, Date startTime, Date endTime) {
        QueryWrapper<HistoryBatteryModule> queryWrapper = new QueryWrapper<>();
        queryWrapper.eq("mac_address", macAddress)
                .orderByAsc("add_time"); // Assuming you want the records ordered by time

        List<HistoryBatteryModule> historyModules = list(queryWrapper);

        System.out.println("SSSHISTOR:" + historyModules.size() + " ST:  " + startTime.toString() + "  ET:  " + endTime.toString());

        if (historyModules.isEmpty()) {
            return CommonResponse.createForError("No records found");
        }

        System.out.println("SSS:" + historyModules.size());
        return CommonResponse.createForSuccess(historyModules);
    }
}
