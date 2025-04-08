package DataAlignmentAndFusionApplication;

import org.mybatis.spring.annotation.MapperScan;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
@MapperScan("DataAlignmentAndFusionApplication.persistance")
public class DataAlignmentAndFusionApplicationPro {
    public static void main(String[] args) {
        SpringApplication.run(DataAlignmentAndFusionApplicationPro.class, args);
    }

}
