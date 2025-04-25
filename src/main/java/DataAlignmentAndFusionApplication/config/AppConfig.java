package DataAlignmentAndFusionApplication.config;

import lombok.Data;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Configuration;
@Data
@Configuration
public class AppConfig {

    @Value("${data.upload-raw-dir}")
    private String uploadRawDir;

    @Value("${data.upload-clean-dir}")
    private String uploadCleanDir;

    @Value("${data.upload-report-dir}")
    private String uploadReportDir;

    @Value("${data.upload-output-dir}")
    private String uploadOutputDir;


}