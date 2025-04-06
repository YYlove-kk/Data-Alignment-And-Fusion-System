package Test.example.batteryScheduling.service.impl;

import Test.example.batteryScheduling.DTO.DeviceDTO;
import Test.example.batteryScheduling.common.CommonResponse;
import Test.example.batteryScheduling.domain.Device;
import Test.example.batteryScheduling.domain.Region;
import Test.example.batteryScheduling.persistance.DeviceMapper;
import Test.example.batteryScheduling.persistance.RegionMapper;
import Test.example.batteryScheduling.service.DeviceService;
import Test.example.batteryScheduling.service.RegionService;
import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.UpdateWrapper;
import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import org.springframework.beans.BeanUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.Date;
import java.util.List;
import java.util.stream.Collectors;

@Validated
@CrossOrigin
@RestController
@Service("regionService")
public class RegionServiceImpl extends ServiceImpl<RegionMapper, Region> implements RegionService {


    @Autowired
    private RegionMapper regionMapper;
    @Autowired
    private DeviceMapper deviceMapper;

    @Override
    public CommonResponse<List<Region>> getAllRegion() {
        // 使用QueryWrapper来构建查询条件
        QueryWrapper<Region> wrapper = new QueryWrapper<>();
        List<Region> regions = list(wrapper);
        if(regions != null && !regions.isEmpty()) {
            // 假设CommonResponse的成功方法是createForSuccess
            return CommonResponse.createForSuccess("获取分区信息成功", regions);
        } else {
            // 假设CommonResponse的失败方法是createForError
            return CommonResponse.createForError("获取分区信息失败!");
        }
    }

    @Override
    public CommonResponse<List<String>> getAllRegionNumber() {
        // 使用 QueryWrapper 来构建查询条件，仅选择不重复的 regionName 字段
        QueryWrapper<Region> queryWrapper = new QueryWrapper<>();
        queryWrapper.select("distinct region_name");

        // 查询所有不重复的regionName
        List<Region> regions = list(queryWrapper);

        if (regions != null && !regions.isEmpty()) {
            // Java 8 流操作，提取regionName属性到新列表
            List<String> regionNames = regions.stream().map(Region::getRegionName).collect(Collectors.toList());
            return CommonResponse.createForSuccess("获取所有分区名称成功", regionNames);
        } else {
            return CommonResponse.createForError("未找到任何分区名称");
        }
    }


    @Override
    public CommonResponse<Void> addRegion(Region region) {
        // 使用QueryWrapper构建查询条件以查找同名分区
        QueryWrapper<Region> regionQueryWrapper = new QueryWrapper<>();
        regionQueryWrapper.eq("region_name", region.getRegionName());

        // 检查设备是否存在
        QueryWrapper<Device> deviceQueryWrapper = new QueryWrapper<>();
        deviceQueryWrapper.eq("device_name", region.getRegionDevice());
        Device device = deviceMapper.selectOne(deviceQueryWrapper);

        if(device == null){
            return CommonResponse.createForError("未找到该设备！");
        }

        // 添加分区
        boolean regionAdded = save(region);
        if (regionAdded) {
//            if (device != null) {
//                // 如果设备存在，更新设备信息
//                device.setDeviceArea(region.getRegionName());
//                int deviceUpdated = deviceMapper.updateById(device);
//                if (deviceUpdated==0) {
//                    return CommonResponse.createForError("分区添加成功，但设备信息更新失败");
//                }
//            } else {
                // 如果设备不存在，仍然创建分区，但返回警告信息
                return CommonResponse.createForSuccessMessage("绑定关系添加成功");
//            }
//            return CommonResponse.createForSuccessMessage("分区及相关设备信息更新成功");
        } else {
            return CommonResponse.createForError("绑定关系添加失败");
        }
    }


    @Override
    public CommonResponse<Void> deleteRegion(Region region) {
        QueryWrapper<Region> queryWrapper = new QueryWrapper<>();
        queryWrapper.eq("region_name", region.getRegionName());
        Region existingRegion = getOne(queryWrapper);

        if (existingRegion != null) {
            // 删除分区
            boolean success = remove(queryWrapper);
//            if (success) {
//                // 更新关联设备的区域信息为“未分配分区”
//                QueryWrapper<Device> deviceQuery = new QueryWrapper<>();
//                deviceQuery.eq("device_area", existingRegion.getRegionName());
//                List<Device> devices = deviceMapper.selectList(deviceQuery);
//                for (Device device : devices) {
//                    device.setDeviceArea("未分配分区");
//                    deviceMapper.updateById(device);
//                }
                return CommonResponse.createForSuccessMessage("分区删除成功");
//            } else {
//                return CommonResponse.createForError("分区删除失败");
//            }
        }
        return CommonResponse.createForError("分区不存在");
    }

    @Override
    public CommonResponse<Void> updateRegion(Region region) {
        QueryWrapper<Region> regionQuery = new QueryWrapper<>();
        regionQuery.eq("region_name", region.getRegionName());
        Region existingRegion = getOne(regionQuery);

        QueryWrapper<Device> deviceQuery = new QueryWrapper<>();
        deviceQuery.eq("device_name", region.getRegionDevice());
        Device device = deviceMapper.selectOne(deviceQuery);

        if(device == null){
            return CommonResponse.createForError("设备不存在！");
        }

        if (existingRegion != null) {
            // 创建UpdateWrapper用于更新分区
            UpdateWrapper<Region> regionUpdate = new UpdateWrapper<>();
            regionUpdate.eq("region_name", existingRegion.getRegionName())
                    .set("region_device", region.getRegionDevice())
                    .set("region_status", region.getRegionStatus())
                    .set("region_describe", region.getRegionDescribe());

            // 更新分区信息
            boolean regionUpdated = update(regionUpdate);
            if (regionUpdated) {
//                // 如果分区信息更新成功，则更新所有相关设备的区域信息
//                deviceQuery = new QueryWrapper<>();
//                deviceQuery.eq("device_area", region.getRegionName());
//
//                //修改分区对应原设备的分区信息
//                device= deviceMapper.selectOne(deviceQuery);
//                device.setDeviceArea("未分配分区");
//                deviceMapper.updateById(device); // 更新设备信息
//
//                //绑定新设备
//                deviceQuery = new QueryWrapper<>();
//                deviceQuery.eq("device_name", region.getRegionDevice());
//                device= deviceMapper.selectOne(deviceQuery);
//                device.setDeviceArea(region.getRegionName());
//                deviceMapper.updateById(device); // 更新设备信息

                return CommonResponse.createForSuccessMessage("分区及相关设备信息更新成功");
            } else {
                return CommonResponse.createForError("分区更新失败");
            }
        }
        return CommonResponse.createForError("分区不存在");
    }




}
