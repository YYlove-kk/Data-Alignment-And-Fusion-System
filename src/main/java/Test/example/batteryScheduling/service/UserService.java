package Test.example.batteryScheduling.service;

import Test.example.batteryScheduling.DTO.UserRegistDTO;
import Test.example.batteryScheduling.common.CommonResponse;
import Test.example.batteryScheduling.domain.User;
import com.baomidou.mybatisplus.extension.service.IService;


public interface UserService extends IService<User> {

    //用户登录
    CommonResponse<User> login(String username, String password);

    User getUserByUserName(String username);

    CommonResponse<User> register(UserRegistDTO userRegisterDto);
}
