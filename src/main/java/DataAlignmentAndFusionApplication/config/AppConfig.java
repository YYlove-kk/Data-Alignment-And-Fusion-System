package DataAlignmentAndFusionApplication.config;

import lombok.Data;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Configuration;
@Data
@Configuration
public class AppConfig {

    @Value("${data.upload-raw-dir}")
    private String uploadDir;

    @Value("${data.upload-report-dir}")
    private String uploadReportDir;

    @Value("${data.upload-clean-dir}")
    private String uploadCleanDir;

    @Value("${algorithm.data-ingest}")
    private String dataIngestPath;

    @Value("${algorithm.image-embed}")
    private String imageEmbedPath;

    @Value("${algorithm.text-time-embed}")
    private String textTimeEmbedPath;

    @Value("${algorithm.interpreter-path}")
    private String interpreterPath;

    @Value("${schema-registry}")
    private String schemaRegistryPath;

    @Value("${data.align-output-dir}")
    private String alignOutputPath;

    @Value("${data.align-source-dir}")
    private String alignSourcePath;

    @Value("${algorithm.train-tcmt}")
    private String alignScriptPath;

    @Value("${algorithm.neo4j-import}")
    private String neo4jScriptPath;

    @Value("${algorithm.knsw_builder}")
    private String knswScriptPath;

    @Value("${algorithm.fusion}")
    private String fusionScriptPath;

    @Value("${algorithm.kpca_reduce}")
    private String kpcaReducePath;
}