
package Test.example.batteryScheduling.persistance;

import Test.example.batteryScheduling.domain.User;
import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import org.springframework.stereotype.Repository;

/**
 * 用户
 */
@Repository
public interface UserMapper extends BaseMapper<User> {

}
