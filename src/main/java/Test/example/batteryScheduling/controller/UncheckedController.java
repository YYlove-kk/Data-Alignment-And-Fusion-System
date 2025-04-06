package Test.example.batteryScheduling.controller;

import Test.example.batteryScheduling.DTO.DeviceDTO;
import Test.example.batteryScheduling.DTO.UncheckedDTO;
import Test.example.batteryScheduling.common.CommonResponse;
import Test.example.batteryScheduling.domain.Device;
import Test.example.batteryScheduling.domain.Unchecked;
import Test.example.batteryScheduling.service.DeviceService;
import Test.example.batteryScheduling.service.UncheckedService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@Validated
@RestController
@RequestMapping("/Unchecked")
public class UncheckedController {
    @Autowired
    private UncheckedService uncheckedService;

    @RequestMapping("/infos")
    public CommonResponse<List<Unchecked>> getAllStock(){
        return uncheckedService.getAllUnchecked();
    }

    @PostMapping("/addStock")
    public CommonResponse<Unchecked> addDevice(@RequestBody UncheckedDTO uncheckedDTODTO) {
        // 这里调用 service 层处理注册逻辑
        return uncheckedService.addUnchecked(uncheckedDTODTO);
    }

//    @PostMapping("/editStock")
//    public CommonResponse<String> editDevice(@RequestBody UncheckedDTO uncheckedDTODTO) {
//        // 这里调用 service 层处理注册逻辑
//        return uncheckedService.editUnchecked(uncheckedDTODTO);
//    }

    @PostMapping("/deleteStock")
    public CommonResponse<String> deleteDevice(@RequestBody UncheckedDTO uncheckedDTODTO) {
        // 这里调用 service 层处理注册逻辑
        return uncheckedService.deleteUnchecked(uncheckedDTODTO);
    }

}
