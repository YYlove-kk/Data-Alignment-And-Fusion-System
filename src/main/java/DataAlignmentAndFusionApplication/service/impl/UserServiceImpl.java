package DataAlignmentAndFusionApplication.service.impl;

import DataAlignmentAndFusionApplication.common.CommonResponse;
import DataAlignmentAndFusionApplication.model.dto.UserRegistDTO;
import DataAlignmentAndFusionApplication.service.UserService;
import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import DataAlignmentAndFusionApplication.model.entity.User;
import DataAlignmentAndFusionApplication.mapper.module.UserMapper;
import org.springframework.beans.BeanUtils;
import org.springframework.stereotype.Service;

import java.util.Date;

/**
* @author 29857
* @description 针对表【user】的数据库操作Service实现
* @createDate 2025-04-22 23:02:16
*/
@Service
public class UserServiceImpl extends ServiceImpl<UserMapper, User>
    implements UserService {
    //登录
    @Override
    public CommonResponse<DataAlignmentAndFusionApplication.model.entity.User> login(String username, String password){

        QueryWrapper<DataAlignmentAndFusionApplication.model.entity.User> queryWrapper = new QueryWrapper<>();
        queryWrapper.eq("username",username);
        DataAlignmentAndFusionApplication.model.entity.User user = getOne(queryWrapper);

        if(user == null){
            return CommonResponse.createForError("用户名不存在");
        }else if(user.getPassword().equals(password)) {
            return CommonResponse.createForSuccess("success",user);
        }else {
            return CommonResponse.createForError("密码错误！");
        }
    }

    @Override
    public DataAlignmentAndFusionApplication.model.entity.User getUserByUserName(String username) {
        QueryWrapper<DataAlignmentAndFusionApplication.model.entity.User> queryWrapper = new QueryWrapper<>();
        queryWrapper.eq("username", username); // 添加条件，用户名字段等于传入的username
        return getOne(queryWrapper); // 使用getOne方法查询满足条件的用户，如果没有找到用户则返回null
    }

    @Override
    public CommonResponse<DataAlignmentAndFusionApplication.model.entity.User> register(UserRegistDTO userRegisterDto) {
        QueryWrapper<DataAlignmentAndFusionApplication.model.entity.User> queryWrapper = new QueryWrapper<>();
        queryWrapper.eq("username", userRegisterDto.getUsername());
        DataAlignmentAndFusionApplication.model.entity.User user = baseMapper.selectOne(queryWrapper);

        if (user != null) {
            return CommonResponse.createForError("用户已存在");
        } else {
            DataAlignmentAndFusionApplication.model.entity.User newUser = new DataAlignmentAndFusionApplication.model.entity.User();
            BeanUtils.copyProperties(userRegisterDto, newUser);
            newUser.setAddtime(new Date()); // 添加注册时间
            // 注意：实际应用中需要对密码进行加密处理，例如使用 BCryptPasswordEncoder
            int result = baseMapper.insert(newUser);
            if (result == 1) {
                return CommonResponse.createForSuccess("注册成功", newUser);
            } else {
                return CommonResponse.createForError("注册失败");
            }
        }
    }
}




