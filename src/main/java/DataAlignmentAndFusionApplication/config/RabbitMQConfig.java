package DataAlignmentAndFusionApplication.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.amqp.core.Queue;

@Configuration
public class RabbitMQConfig {

    @Value("${mq.upload-to-cleaning}")
    private String uploadToCleaningQueue;

    @Value("${mq.cleaning-to-database}")
    private String cleaningToDatabaseQueue;

    @Bean
    public Queue uploadToCleaningQueue() {
        return new Queue(uploadToCleaningQueue, true);
    }

    @Bean
    public Queue cleaningToDatabaseQueue() {
        return new Queue(cleaningToDatabaseQueue, true);
    }

}
