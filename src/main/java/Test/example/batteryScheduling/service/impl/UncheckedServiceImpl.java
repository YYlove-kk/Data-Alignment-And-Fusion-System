package Test.example.batteryScheduling.service.impl;

import Test.example.batteryScheduling.DTO.UncheckedDTO;
import Test.example.batteryScheduling.common.CommonResponse;
import Test.example.batteryScheduling.domain.Unchecked;
import Test.example.batteryScheduling.persistance.UncheckedMapper;
import Test.example.batteryScheduling.persistance.UncheckedMapper;
import Test.example.batteryScheduling.service.UncheckedService;
import Test.example.batteryScheduling.service.UncheckedService;
import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.UpdateWrapper;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import org.springframework.beans.BeanUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service("uncheckedService")
public class UncheckedServiceImpl extends ServiceImpl<UncheckedMapper, Unchecked> implements UncheckedService {

    @Autowired
    private UncheckedMapper uncheckedMapper;

    @Override
    public CommonResponse<List<Unchecked>> getAllUnchecked() {
        List<Unchecked> unchecked = uncheckedMapper.selectList(null);
        return CommonResponse.createForSuccess(unchecked);
    }

    @Override
    public void updateUnchecked(Unchecked unchecked) {
        uncheckedMapper.updateById(unchecked);
    }

    @Override
    public CommonResponse<Unchecked> addUnchecked(UncheckedDTO uncheckedDTO) {
        Unchecked unchecked = new Unchecked();
        BeanUtils.copyProperties(uncheckedDTO, unchecked);
        int result = uncheckedMapper.insert(unchecked);
        if (result == 1) {
            return CommonResponse.createForSuccess("添加库存成功",unchecked);
        } else {
            return CommonResponse.createForError("添加库存失败");
        }
    }

    @Override
    public CommonResponse<String> deleteUnchecked(UncheckedDTO uncheckedDTO) {
        QueryWrapper<Unchecked> queryWrapper = new QueryWrapper<>();
        queryWrapper.eq("set_number", uncheckedDTO.getSetNumber());
        int result = uncheckedMapper.delete(queryWrapper);
        if (result == 1) {
            return CommonResponse.createForSuccess("删除库存成功");
        } else {
            return CommonResponse.createForError("删除库存失败");
        }
    }

//    public CommonResponse<String> editUnchecked(UncheckedDTO uncheckedDTO) {
//        UpdateWrapper<Unchecked> updateWrapper = new UpdateWrapper<>();
//        updateWrapper.eq("set_number", uncheckedDTO.getSetNumber()); // 确保DTO包含id
//
//        // 只有在DTO中相应字段不为空时才设置更新，避免覆盖已有数据
//        if (uncheckedDTO.getSetNumber() != null) {
//            updateWrapper.set("set_number", uncheckedDTO.getSetNumber());
//        }
//        if (uncheckedDTO.getSetStatus() != null) {
//            updateWrapper.set("set_status", uncheckedDTO.getSetStatus());
//        }
//        if (uncheckedDTO.getSetReason() != null) {
//            updateWrapper.set("set_reason", uncheckedDTO.getSetReason());
//        }
//        if (uncheckedDTO.getAdd_time() != null) {
//            updateWrapper.set("add_time", uncheckedDTO.getAdd_time());
//        }
//
//        // 执行更新操作
//        boolean result = uncheckedMapper.update(null, updateWrapper) > 0;
//        if (result) {
//            return CommonResponse.createForSuccess("库存更新成功");
//        } else {
//            return CommonResponse.createForError("库存更新失败");
//        }
//    }
}
