package DataAlignmentAndFusionApplication.controller;

import DataAlignmentAndFusionApplication.config.AppConfig;
import DataAlignmentAndFusionApplication.model.dto.AlignmentRequest;
import DataAlignmentAndFusionApplication.model.entity.AlignmentRecord;
import DataAlignmentAndFusionApplication.model.entity.AlignmentResult;
import DataAlignmentAndFusionApplication.service.AlignmentService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/alignment")
public class AlignmentController {

    private final AlignmentService alignmentService;

    @Autowired
    public AlignmentController(AlignmentService alignmentService) {
        this.alignmentService = alignmentService;
    }

    @PostMapping("/align")
    public AlignmentResult align() {
        try {
            // 调用服务层方法进行对齐操作
            return alignmentService.alignTextAndImage();
        } catch (Exception e) {
            e.printStackTrace();
            // 可以返回错误信息或其他处理
            return null;
        }
    }

    @GetMapping("/list")
    public List<AlignmentRecord> listAll() {
        return alignmentService.getAllResults();
    }
}
