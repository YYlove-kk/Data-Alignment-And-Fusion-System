package Test.example.batteryScheduling.controller;

import Test.example.batteryScheduling.DTO.DeviceDTO;
import Test.example.batteryScheduling.common.CommonResponse;
import Test.example.batteryScheduling.domain.Device;
import Test.example.batteryScheduling.domain.Region;
import Test.example.batteryScheduling.service.DeviceService;
import Test.example.batteryScheduling.service.RegionService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@Validated
@RestController
@RequestMapping("/Region")
public class RegionController {
    @Autowired
    RegionService regionService;

    @RequestMapping("/infos")
    public CommonResponse<List<Region>> getAllRegion(){
        return regionService.getAllRegion();
    }

    @RequestMapping("/addRegion")
    public CommonResponse<Void> addRegion(@RequestBody Region newRegion){
        return regionService.addRegion(newRegion);
    }

    @RequestMapping("/getAllRegionNumber")
    public CommonResponse<List<String>> getAllRegionNumber(){
        return regionService.getAllRegionNumber();
    }

    @RequestMapping("/deleteRegion")
    public CommonResponse<Void> deleteRegion(@RequestBody Region newRegion){
        return regionService.deleteRegion(newRegion);
    }

    @RequestMapping("/updateRegion")
    public CommonResponse<Void> updateRegion(@RequestBody Region newRegion){
        return regionService.updateRegion(newRegion);
    }
}
