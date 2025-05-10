package DataAlignmentAndFusionApplication.common;

import org.springframework.stereotype.Component;

import java.util.concurrent.atomic.AtomicInteger;

@Component
public class TagGenerator {
    private static final AtomicInteger tagCounter = new AtomicInteger(0);

    public int generateTag() {
        return tagCounter.getAndIncrement();
    }
}