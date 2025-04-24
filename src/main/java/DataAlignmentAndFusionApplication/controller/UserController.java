package DataAlignmentAndFusionApplication.controller;


import DataAlignmentAndFusionApplication.model.dto.UserLoginDTO;
import DataAlignmentAndFusionApplication.model.dto.UserRegistDTO;
import DataAlignmentAndFusionApplication.common.CommonResp;
import DataAlignmentAndFusionApplication.model.entity.User;
import DataAlignmentAndFusionApplication.service.UserService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;

@Validated
@CrossOrigin
@RestController
@RequestMapping("/api/users")
public class UserController {
	@Autowired
	private UserService userService;

	//  登录
	@PostMapping("/login")
	public CommonResp<User> login(@RequestBody UserLoginDTO userLoginDto){
		System.out.println("Login:" + userLoginDto.getUsername() + "  PWD: " + userLoginDto.getPassword());
		return userService.login(userLoginDto.getUsername(), userLoginDto.getPassword());
	}

	// 注册
	@PostMapping("/register")
	public CommonResp<User> register(@RequestBody UserRegistDTO userRegisterDto) {
		System.out.println("RG： " + userRegisterDto.toString());
		// 这里调用 service 层处理注册逻辑
		return userService.register(userRegisterDto);
	}
}
