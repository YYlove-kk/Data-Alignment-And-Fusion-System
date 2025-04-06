package Test.example.batteryScheduling.service.impl;

import Test.example.batteryScheduling.DTO.BatteryModuleDTO;
import Test.example.batteryScheduling.common.CommonResponse;
import Test.example.batteryScheduling.domain.BatteryModule;
import Test.example.batteryScheduling.domain.Device;
import Test.example.batteryScheduling.persistance.BatteryModuleMapper;
import Test.example.batteryScheduling.service.BatteryModuleService;
import Test.example.batteryScheduling.service.DeviceService;
import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import org.springframework.beans.BeanUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.text.SimpleDateFormat;
import java.util.*;
import java.util.stream.Collectors;

@Validated
@CrossOrigin
@RestController
@RequestMapping("/BatteryModule")
@Service("batteryModuleService")
public class BatteryModuleServiceImpl extends ServiceImpl<BatteryModuleMapper, BatteryModule> implements BatteryModuleService {
    @Autowired
    BatteryModuleMapper batteryModuleMapper;
    @Autowired
    private DeviceService deviceService;
    @Override
    public CommonResponse<List<BatteryModuleDTO>> getAllBatteryModule() {
        QueryWrapper<BatteryModule> wrapper = new QueryWrapper<>();
        // 修改查询条件，包括entry_status为1或2的记录
        wrapper.in("entry_status", Arrays.asList("1", "2"));
        System.out.println("Search All Module");
        List<BatteryModule> batteryModuleList = list(wrapper);
        if (batteryModuleList != null && !batteryModuleList.isEmpty()) {
            List<BatteryModuleDTO> batteryModuleDTOList = batteryModuleList.stream().map(batteryModule -> {
                BatteryModuleDTO dto = new BatteryModuleDTO();
                BeanUtils.copyProperties(batteryModule, dto);
                // 特别处理id，将其转换为String
                dto.setId(String.valueOf(batteryModule.getId()));
                return dto;
            }).collect(Collectors.toList());

            System.out.println("可用电池模组数量：" + batteryModuleDTOList.size());
            // 仅用于日志，显示第一个DTO的ID，确认其为String类型
            System.out.println("FID: " + batteryModuleDTOList.get(0).getId());
            return CommonResponse.createForSuccess("success", batteryModuleDTOList);
        } else {
            return CommonResponse.createForError("没有找到可用的电池模组");
        }
    }

    @Override
    public CommonResponse<Map<String, Object>> getPaginatedBatteryModules(int page, int size) {
        Page<BatteryModule> pager = new Page<>(page, size);
        QueryWrapper<BatteryModule> wrapper = new QueryWrapper<>();
        wrapper.orderByDesc("add_time"); // 按照 'add_time' 降序排序
        IPage<BatteryModule> result = this.page(pager, wrapper);

        List<BatteryModuleDTO> batteryModuleDTOs = result.getRecords().stream().map(batteryModule -> {
            BatteryModuleDTO dto = new BatteryModuleDTO();
            BeanUtils.copyProperties(batteryModule, dto);
            return dto;
        }).collect(Collectors.toList());

        Map<String, Object> response = new HashMap<>();
        response.put("data", batteryModuleDTOs);
        response.put("currentPage", result.getCurrent());
        response.put("totalPages", result.getPages());
        response.put("totalItems", result.getTotal());
        System.out.println("当前页：" + result.getCurrent());
        System.out.println("总页数：" + result.getPages());
        System.out.println("总记录数：" + result.getTotal());
        System.out.println("本页记录数：" + batteryModuleDTOs.size());


        return CommonResponse.createForSuccess("分页数据获取成功", response);
    }


    @Override
    public CommonResponse<Void> batchAddBatteryModules(List<BatteryModuleDTO> batteryModuleDTOs) {
        System.out.println("进入批量添加或更新电池模组");
        List<BatteryModule> toAdd = new ArrayList<>();
        List<BatteryModule> toUpdate = new ArrayList<>();

        SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
        dateFormat.setTimeZone(TimeZone.getTimeZone("Asia/Shanghai")); // 设置时区为北京时间

        // 获取当前北京时间
        Date now = new Date();
        System.out.println("Now1: " + now);

        String formattedDate = dateFormat.format(now);

        System.out.println("Now2: " + now);

        System.out.println("Now2: " + formattedDate);

        // 检查每个模组是否已存在
        batteryModuleDTOs.forEach(dto -> {
            QueryWrapper<BatteryModule> wrapper = new QueryWrapper<>();
            wrapper.eq("mac_address", dto.getMacAddress());
            BatteryModule existingModule = batteryModuleMapper.selectOne(wrapper);
            BatteryModule entity = new BatteryModule();
            BeanUtils.copyProperties(dto, entity);
            entity.setEntryStatus("1");  // 设置为已入库
            entity.setAdd_time(now); // 设置当前时间为添加或更新时间

            if (existingModule == null) {
                toAdd.add(entity);
                System.out.println("ADD:" + entity.getPartNumber());
            } else {
                entity.setId(existingModule.getId()); // 确保更新而不是创建一个新的记录
                toUpdate.add(entity);
                System.out.println("Update:" + existingModule.getPartNumber());
            }
        });

        // 统计每个区域新增或更新的模组数量
        Map<String, Long> moduleDivideCount = batteryModuleDTOs.stream()
                .collect(Collectors.groupingBy(BatteryModuleDTO::getModule_divide, Collectors.counting()));

        try {
            boolean saved;
            if(toUpdate.isEmpty() && toAdd.isEmpty()){
                return CommonResponse.createForError("无有效数据");
            }else if(toAdd.isEmpty()){
                saved = saveOrUpdateBatch(toUpdate);
            }else if(toUpdate.isEmpty()){
                saved = saveOrUpdateBatch(toAdd);
            }else{
                saved = saveOrUpdateBatch(toAdd) && saveOrUpdateBatch(toUpdate); // 保存或更新记录
            }

            if (saved) {
                return CommonResponse.createForSuccessMessage("批量添加或更新成功");
            } else {
                // 处理不成功的逻辑...
                return CommonResponse.createForError("批量添加或更新部分失败");
            }
        } catch (Exception e) {
            System.out.println("批量添加或更新出错: " + e.getMessage());
            e.printStackTrace(); // 打印完整的堆栈跟踪
            return CommonResponse.createForError("批量添加或更新失败: " + e.getMessage());
        }
    }

    public CommonResponse<Void> batchSoftDeleteBatteryModules(List<String> ids) {
        // 将String类型的id列表转换为Long类型的id列表
        List<Long> longIds = ids.stream().map(Long::parseLong).collect(Collectors.toList());
        System.out.println("SOFT DELETE:  " + ids.toString());

        // 获取所有要更新的BatteryModule实体
        List<BatteryModule> batteryModules = batteryModuleMapper.selectBatchIds(longIds);
        boolean allUpdated = true;

        for (BatteryModule batteryModule : batteryModules) {
            // 设置entry_status为2表示软删除
            batteryModule.setEntryStatus("2");
            int updateResult = batteryModuleMapper.updateById(batteryModule);
            if (updateResult == 0) {
                allUpdated = false;  // 如果任何一次更新失败，则标记为失败
            }
//
//            // 更新设备监控模组数量
//            if (batteryModule != null) {
//                String moduleDivide = batteryModule.getModule_divide();
//                Device device = deviceService.getDeviceByArea(moduleDivide);
//                if (device != null && device.getDeviceModuleAccount() > 0) {
//                    device.setDeviceModuleAccount(device.getDeviceModuleAccount() - 1);
//                    deviceService.updateDevice(device);
//                }
//            }
        }

        // 根据操作结果返回相应的通用响应
        if (allUpdated) {
            return CommonResponse.createForSuccessMessage("批量删除成功");
        } else {
            return CommonResponse.createForError("批量删除部分失败");
        }
    }




    @Override
    @Transactional
    public CommonResponse<Void> updateBatteryModuleStatus() {
        // 构建删除条件
        QueryWrapper<BatteryModule> deleteCondition = new QueryWrapper<>();
        deleteCondition.le("module_soc", 20)
                .or()
                .le("module_temperature", 0).ge("module_temperature", -10)
                .or()
                .between("module_temperature", 35, 40);
        try {
            boolean success = remove(deleteCondition); // 执行删除操作

            if (success) {
                // 如果删除成功
                return CommonResponse.createForSuccessMessage("设备状态更新成功，符合条件的设备已删除");
            } else {
                // 如果删除失败，可能是因为没有找到符合条件的记录
                return CommonResponse.createForError("设备状态更新失败，未找到符合条件的设备或删除操作未执行");
            }
        } catch (Exception e) {
            // 捕获到异常，可能是数据库操作异常
            // 记录日志 e.printStackTrace();
            return CommonResponse.createForError("设备状态更新过程中出现异常：" + e.getMessage());
        }
    }

    @Override
    public BatteryModule selectBatteryModuleByMac(String macAddress) {
        // 设置查询条件
        QueryWrapper<BatteryModule> queryWrapper = new QueryWrapper<>();
        queryWrapper.eq("mac_address", macAddress)
                .orderByDesc("add_time")
                .last("limit 1"); // 只获取最新的一条记录

        // 查询特定MAC地址的最新一条记录
        return batteryModuleMapper.selectOne(queryWrapper);
    }

}
