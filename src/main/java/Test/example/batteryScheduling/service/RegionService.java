package Test.example.batteryScheduling.service;

import Test.example.batteryScheduling.DTO.DeviceDTO;
import Test.example.batteryScheduling.common.CommonResponse;
import Test.example.batteryScheduling.domain.Device;
import Test.example.batteryScheduling.domain.Region;
import com.baomidou.mybatisplus.extension.service.IService;

import java.util.List;

public interface RegionService extends IService<Region> {
    CommonResponse<List<Region>> getAllRegion();
    CommonResponse<List<String>> getAllRegionNumber();

    CommonResponse<Void> addRegion(Region region);

    CommonResponse<Void> deleteRegion(Region newRegion);

    CommonResponse<Void> updateRegion(Region newRegion);
}
