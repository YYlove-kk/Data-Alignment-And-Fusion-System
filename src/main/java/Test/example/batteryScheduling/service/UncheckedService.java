package Test.example.batteryScheduling.service;

import Test.example.batteryScheduling.DTO.DeviceDTO;
import Test.example.batteryScheduling.DTO.UncheckedDTO;
import Test.example.batteryScheduling.common.CommonResponse;
import Test.example.batteryScheduling.domain.Device;
import Test.example.batteryScheduling.domain.Unchecked;
import com.baomidou.mybatisplus.extension.service.IService;

import java.util.List;

public interface UncheckedService extends IService<Unchecked> {
    CommonResponse<List<Unchecked>> getAllUnchecked();

    void updateUnchecked(Unchecked unchecked);

    CommonResponse<Unchecked> addUnchecked(UncheckedDTO uncheckedDTO);
    CommonResponse<String> deleteUnchecked(UncheckedDTO uncheckedDTO);

//    CommonResponse<String> editUnchecked(UncheckedDTO uncheckedDTO);
}
