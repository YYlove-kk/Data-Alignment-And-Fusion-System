package Test.example.batteryScheduling.controller;

import Test.example.batteryScheduling.DTO.AdminLoginDTO;
import Test.example.batteryScheduling.common.CommonResponse;
import Test.example.batteryScheduling.domain.Admin;
import Test.example.batteryScheduling.service.AdminService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;

@Validated
@CrossOrigin
@RestController
@RequestMapping("/admin")
public class AdminController {
    @Autowired
    private AdminService adminService;

    //  登录
    @PostMapping("/login")
    public CommonResponse<Admin> login(@RequestBody AdminLoginDTO adminLoginDTO){
        System.out.println("LoginADMIN:" + adminLoginDTO.getUsername() + "  PWD: " + adminLoginDTO.getPassword());
        return adminService.login(adminLoginDTO.getUsername(), adminLoginDTO.getPassword());
    }
}
