package Test.example.batteryScheduling.common;

import Test.example.batteryScheduling.domain.BatteryCell;
import Test.example.batteryScheduling.domain.BatteryModule;
import Test.example.batteryScheduling.domain.HistoryBatteryModule;
import Test.example.batteryScheduling.persistance.HistoryBatteryModuleMapper;
import Test.example.batteryScheduling.service.BatteryCellService;
import Test.example.batteryScheduling.service.BatteryModuleService;
import Test.example.batteryScheduling.service.impl.BatteryModuleServiceImpl;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.junit4.SpringRunner;

import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.*;

@RunWith(SpringRunner.class)
@SpringBootTest
public class UnitTestClass {

    @Autowired
    private BatteryModuleService batteryModuleService;
    @Autowired
    private BatteryCellService batteryCellService;
    @Autowired
    private HistoryBatteryModuleMapper historyBatteryModuleMapper;

    @Test
    public void generateTestData() {
        String macAddress = "MAC-3fe8eea4f26414b5";
        int battertNumber = 8;
        int times = 5; // 自定义往前推的周数

        BatteryModule latestModuleInfo = batteryModuleService.selectBatteryModuleByMac(macAddress);

        //插入电芯数据
        insertBatteryCellData(macAddress,battertNumber);

        //插入历史信息数据
//        if (latestModuleInfo != null) {
//            // 以最新记录的add_time作为终止时间点endTime
//            Calendar cal = Calendar.getInstance();
//            cal.setTime(latestModuleInfo.getAdd_time());
//            Date endTime = cal.getTime();
//
//            // 计算startTime
//            cal.add(Calendar.WEEK_OF_YEAR, -times);
//            Date startTime = cal.getTime();
//
//            // 计算需要插入的数据条数
//            long weeksBetween = (endTime.getTime() - startTime.getTime()) / (1000 * 60 * 60 * 24 * 7);
//
//            for (int i = 0; i < weeksBetween; i++) {
//                // 为每周生成一个时间点
//                Calendar weekCal = Calendar.getInstance();
//                weekCal.setTime(startTime);
//                weekCal.add(Calendar.WEEK_OF_YEAR, i);
//
//                // 生成随机SOC和温度数据
//                double moduleSoc = new Random().nextDouble() * 100; // Range [0,100]
//                double moduleTemperature = -10 + new Random().nextDouble() * 50; // Range [-10,40]
//
//                // 每周生成并插入一条数据
//                insertHistoryData(weekCal.getTime(), macAddress, latestModuleInfo.getPartNumber(),
//                        latestModuleInfo.getSetNumber(), latestModuleInfo.getModuleName(),
//                        latestModuleInfo.getWilVersion(), latestModuleInfo.getScriptVersion(),
//                        latestModuleInfo.getEntryStatus(), moduleSoc, moduleTemperature);
//            }
//        } else {
//            System.out.println("No BatteryModule info found for the given MAC address.");
//        }
    }

    public void insertHistoryData(Date timestamp, String macAddress, String partNumber, String setNumber,
                                  String moduleName, String wilVersion, String scriptVersion, String entryStatus,
                                  double moduleSoc, double moduleTemperature) {

        HistoryBatteryModule module = new HistoryBatteryModule();

        // Set custom fields
        module.setMacAddress(macAddress);
        module.setPartNumber(partNumber);
        module.setSetNumber(setNumber);
        module.setModuleName(moduleName);
        module.setWilVersion(wilVersion);
        module.setScriptVersion(scriptVersion);
        module.setEntryStatus(entryStatus);

        // Set generated moduleSoc and moduleTemperature
        module.setModuleSoc(moduleSoc);
        module.setModuleTemperature(moduleTemperature);

        // Set add_time for each week
        module.setAdd_time(timestamp);

        boolean save = historyBatteryModuleMapper.insert(module) > 0;
        if(save){
            System.out.println("Inserted History Module Info successfully for the week of: " + timestamp);
        } else {
            System.out.println("Failed to insert History Module Info for the week of: " + timestamp);
        }
    }

    public void insertBatteryCellData(String macAddress, int batteryNumber) {
        List<BatteryCell> cells = new ArrayList<>();
        Random random = new Random();

        for (int i = 0; i < batteryNumber; i++) {
            BatteryCell cell = new BatteryCell();
            cell.setMacAddress(macAddress);
            cell.setBatteryCellNumber("Cell" + (i + 1));
            double batteryVolt = 1 + (4 * random.nextDouble()); // 生成1~5之间的随机double值
            cell.setBatteryVolt(batteryVolt);

            cells.add(cell);
        }

        // 批量插入数据库
        boolean result = batteryCellService.saveBatch(cells);

        // 打印插入结果
        System.out.println("Batch insertion result: " + result);
    }

    public void DataFormatTest() throws ParseException {
        SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
        // 创建一个Date对象，它代表当前的日期和时间
        Date now = new Date();
        System.out.println(now.toString());
        Date ee = dateFormat.parse(now.toString());
        System.out.println(ee.toString());
    }
}
