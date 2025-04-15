package DataAlignmentAndFusionApplication.service;

import DataAlignmentAndFusionApplication.model.dto.UserRegistDTO;
import DataAlignmentAndFusionApplication.common.CommonResponse;
import DataAlignmentAndFusionApplication.model.entity.User;
import com.baomidou.mybatisplus.extension.service.IService;


public interface UserService extends IService<User> {

    //用户登录
    CommonResponse<User> login(String username, String password);

    User getUserByUserName(String username);

    CommonResponse<User> register(UserRegistDTO userRegisterDto);
}
