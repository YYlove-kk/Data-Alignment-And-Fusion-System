package Test.example.batteryScheduling.controller;

import Test.example.batteryScheduling.DTO.DeviceDTO;
import Test.example.batteryScheduling.DTO.UserRegistDTO;
import Test.example.batteryScheduling.common.CommonResponse;
import Test.example.batteryScheduling.domain.Device;
import Test.example.batteryScheduling.domain.User;
import Test.example.batteryScheduling.service.DeviceService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@Validated
@RestController
@RequestMapping("/Device")
public class DeviceController {
    @Autowired
    private DeviceService deviceService;

    @RequestMapping("/infos")
    public CommonResponse<List<Device>> getAllDevice(){
        return deviceService.getAllDevice();
    }

    @PostMapping("/addDevice")
    public CommonResponse<Device> addDevice(@RequestBody DeviceDTO deviceDTO) {
        System.out.println("RG： " + deviceDTO.getDeviceName());
        // 这里调用 service 层处理注册逻辑
        return deviceService.addDevice(deviceDTO);
    }

    @PostMapping("/editDevice")
    public CommonResponse<String> editDevice(@RequestBody DeviceDTO deviceDTO) {
        System.out.println("RG： " + deviceDTO.getDeviceName());
        // 这里调用 service 层处理注册逻辑
        return deviceService.editDevice(deviceDTO);
    }

    @PostMapping("/deleteDevice")
    public CommonResponse<String> deleteDevice(@RequestBody DeviceDTO deviceDTO) {
        System.out.println("DG： " + deviceDTO.getDeviceName());
        // 这里调用 service 层处理注册逻辑
        return deviceService.deleteDevice(deviceDTO);
    }

}
