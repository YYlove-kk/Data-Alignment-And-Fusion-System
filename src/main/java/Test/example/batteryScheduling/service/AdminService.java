package Test.example.batteryScheduling.service;

import Test.example.batteryScheduling.common.CommonResponse;
import com.baomidou.mybatisplus.extension.service.IService;
import Test.example.batteryScheduling.domain.Admin;


public interface AdminService extends IService<Admin> {

    //用户登录
    CommonResponse<Admin> login(String username, String password);
}
