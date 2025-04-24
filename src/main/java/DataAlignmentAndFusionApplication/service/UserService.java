package DataAlignmentAndFusionApplication.service;

import DataAlignmentAndFusionApplication.common.CommonResp;
import DataAlignmentAndFusionApplication.model.dto.UserRegistDTO;
import DataAlignmentAndFusionApplication.model.entity.User;
import com.baomidou.mybatisplus.extension.service.IService;

/**
* @author 29857
* @description 针对表【user】的数据库操作Service
* @createDate 2025-04-22 23:02:16
*/
public interface UserService extends IService<User> {
    //用户登录
    CommonResp<User> login(String username, String password);

    DataAlignmentAndFusionApplication.model.entity.User getUserByUserName(String username);

    CommonResp<User> register(UserRegistDTO userRegisterDto);
}
