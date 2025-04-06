package Test.example.batteryScheduling.service;

import Test.example.batteryScheduling.DTO.DeviceDTO;
import Test.example.batteryScheduling.DTO.UserRegistDTO;
import Test.example.batteryScheduling.common.CommonResponse;
import Test.example.batteryScheduling.domain.Device;
import Test.example.batteryScheduling.domain.User;
import com.baomidou.mybatisplus.extension.service.IService;

import java.util.List;

public interface DeviceService extends IService<Device> {
    CommonResponse<List<Device>> getAllDevice();
    Device getDeviceByArea(String area);
    void updateDevice(Device device);

    CommonResponse<Device> addDevice(DeviceDTO deviceDTO);
    CommonResponse<String> deleteDevice(DeviceDTO deviceDTO);

    CommonResponse<String> editDevice(DeviceDTO deviceDTO);
}
