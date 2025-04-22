package DataAlignmentAndFusionApplication.service;

import DataAlignmentAndFusionApplication.common.CommonResponse;
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
    CommonResponse<DataAlignmentAndFusionApplication.model.entity.User> login(String username, String password);

    DataAlignmentAndFusionApplication.model.entity.User getUserByUserName(String username);

    CommonResponse<DataAlignmentAndFusionApplication.model.entity.User> register(UserRegistDTO userRegisterDto);
}
