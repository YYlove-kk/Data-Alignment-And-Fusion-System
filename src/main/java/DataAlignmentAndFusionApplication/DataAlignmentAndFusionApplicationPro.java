package DataAlignmentAndFusionApplication;

import DataAlignmentAndFusionApplication.model.vo.GraphVO;
import DataAlignmentAndFusionApplication.util.GraphQueryUtil;
import org.mybatis.spring.annotation.MapperScan;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
@MapperScan("DataAlignmentAndFusionApplication.mapper")
public class DataAlignmentAndFusionApplicationPro {
    public static void main(String[] args) {
        SpringApplication.run(DataAlignmentAndFusionApplicationPro.class, args);
    }

}


