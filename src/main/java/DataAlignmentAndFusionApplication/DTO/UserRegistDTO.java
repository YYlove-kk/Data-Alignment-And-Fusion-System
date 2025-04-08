package DataAlignmentAndFusionApplication.DTO;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import lombok.Getter;

import java.util.Date;

@Getter
public class UserRegistDTO {


    /**
     * 用户账号
     */
    private String username;
    /**
     * 密码
     */
    private String password;


    private String email;

    public void setUsername(String username) {
        this.username = username;
    }

    public void setPassword(String password) {
        this.password = password;
    }

    public void setEmail(String email) {
        this.email = email;
    }
}
