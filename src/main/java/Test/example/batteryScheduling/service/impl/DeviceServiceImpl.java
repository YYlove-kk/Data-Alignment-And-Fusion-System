package Test.example.batteryScheduling.service.impl;

import Test.example.batteryScheduling.DTO.DeviceDTO;
import Test.example.batteryScheduling.DTO.UserRegistDTO;
import Test.example.batteryScheduling.common.CommonResponse;
import Test.example.batteryScheduling.domain.Device;
import Test.example.batteryScheduling.domain.Region;
import Test.example.batteryScheduling.domain.User;
import Test.example.batteryScheduling.persistance.RegionMapper;
import Test.example.batteryScheduling.service.DeviceService;
import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.UpdateWrapper;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import Test.example.batteryScheduling.persistance.DeviceMapper;
import org.springframework.beans.BeanUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.Date;
import java.util.List;

@Validated
@CrossOrigin
@RestController
@RequestMapping("/Device")
@Service("deviceService")
public class DeviceServiceImpl extends ServiceImpl<DeviceMapper, Device> implements DeviceService {

    @Autowired
    RegionMapper regionMapper;

    @Override
    public CommonResponse<List<Device>> getAllDevice() {
        // 使用QueryWrapper来构建查询条件
        QueryWrapper<Device> wrapper = new QueryWrapper<>();
        List<Device> deviceList = list(wrapper);
        if(deviceList != null && !deviceList.isEmpty()) {
            // 假设CommonResponse的成功方法是createForSuccess
            return CommonResponse.createForSuccess("success", deviceList);
        } else {
            // 假设CommonResponse的失败方法是createForError
            return CommonResponse.createForError("恒温房设备数为0");
        }
    }

    public CommonResponse<Device> addDevice(DeviceDTO deviceDTO) {
        QueryWrapper<Device> queryWrapper = new QueryWrapper<>();
        queryWrapper.eq("device_name", deviceDTO.getDeviceName());
        Device device = baseMapper.selectOne(queryWrapper);

        if (device != null) {
            return CommonResponse.createForError("设备已存在");
        } else {
            Device newDevice = new Device();
            BeanUtils.copyProperties(deviceDTO, newDevice);
            newDevice.setDeviceAddtime(new Date()); // 添加注册时间
            // 注意：实际应用中需要对密码进行加密处理，例如使用 BCryptPasswordEncoder
            int result = baseMapper.insert(newDevice);
            if (result == 1) {
                return CommonResponse.createForSuccess("设备添加成功", newDevice);
            } else {
                return CommonResponse.createForError("设备添加失败");
            }
        }
    }

    @Override
    public CommonResponse<String> deleteDevice(DeviceDTO deviceDTO) {
        QueryWrapper<Device> queryWrapper = new QueryWrapper<>();
        queryWrapper.eq("device_name", deviceDTO.getDeviceName());
        Device device = baseMapper.selectOne(queryWrapper);

        if (device != null) {
//            QueryWrapper<Region> regionWrapper = new QueryWrapper<>();
//            regionWrapper.eq("region_name", deviceDTO.getDeviceArea());
//            Region region = regionMapper.selectOne(regionWrapper);
//            if(region != null){
//                UpdateWrapper<Region> regionUpdate = new UpdateWrapper<>();
//                regionUpdate.eq("region_name", region.getRegionName())
//                        .set("region_device", "未分配设备")
//                        .set("region_status", region.getRegionStatus())
//                        .set("region_describe", region.getRegionDescribe());
//                // 更新分区信息
//                int regionUpdated = regionMapper.update(region,regionUpdate);
//                if(regionUpdated == 0){
//                    return CommonResponse.createForError("关联分区更新失败");
//                }
//            }

            int result = baseMapper.delete(queryWrapper);
            return CommonResponse.createForSuccess("设备删除成功");
        } else {
            return CommonResponse.createForError("设备添加失败");
        }
    }

    @Override
    public CommonResponse<String> editDevice(DeviceDTO deviceDTO) {
        QueryWrapper<Device> queryWrapper = new QueryWrapper<>();
        queryWrapper.eq("device_name", deviceDTO.getDeviceName());
        Device device = baseMapper.selectOne(queryWrapper);

        if(device != null){
            BeanUtils.copyProperties(deviceDTO,device);
            device.setDeviceStatus("重启");
            updateDevice(device);
            return CommonResponse.createForSuccess("设备修改成功");
        }else{
            return CommonResponse.createForError("设备修改失败");
        }
    }

    @Override
    public Device getDeviceByArea(String area) {
        // 使用 QueryWrapper 来根据区域查询设备
        QueryWrapper<Device> queryWrapper = new QueryWrapper<>();
        queryWrapper.eq("device_Area", area);
        // 使用 getOne 方法获取匹配的设备（如果有多条匹配数据，需设置 `limit 1`）
        return getOne(queryWrapper, false);
    }

    @Override
    public void updateDevice(Device device) {
        // 直接通过 `updateById` 更新设备信息
        updateById(device);
    }
}
