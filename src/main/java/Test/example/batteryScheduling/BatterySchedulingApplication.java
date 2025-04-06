package Test.example.batteryScheduling;

import org.mybatis.spring.annotation.MapperScan;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
@MapperScan("Test.example.batteryScheduling.persistance")
public class BatterySchedulingApplication {
    public static void main(String[] args) {
        SpringApplication.run(BatterySchedulingApplication.class, args);
    }

}
