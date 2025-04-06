package Test.example.batteryScheduling.service.impl;

import Test.example.batteryScheduling.common.CommonResponse;
import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import Test.example.batteryScheduling.domain.Admin;
import Test.example.batteryScheduling.persistance.AdminMapper;
import Test.example.batteryScheduling.service.AdminService;
import org.springframework.stereotype.Service;

@Service("adminService")
public class AdminServiceImpl extends ServiceImpl<AdminMapper, Admin> implements AdminService {


    //登录
    @Override
    public CommonResponse<Admin> login(String username, String password){

        QueryWrapper<Admin> queryWrapper = new QueryWrapper<>();
        queryWrapper.eq("username",username);
        Admin admin = getOne(queryWrapper);

        if(admin == null){
            return CommonResponse.createForError("用户名不存在");
        }else if(admin.getPassword().equals(password)) {
            return CommonResponse.createForSuccess("success",admin);
        }else {
            return CommonResponse.createForError("密码错误！");
        }
    }

}
