package DataAlignmentAndFusionApplication.controller;

import DataAlignmentAndFusionApplication.model.vo.AlignmentVO;
import DataAlignmentAndFusionApplication.service.AlignmentService;
import DataAlignmentAndFusionApplication.util.Result;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/alignment")
public class AlignmentController {
    @Autowired
    private AlignmentService alignmentService;


    @PostMapping("/align")
    public Result<String> align() {
        try {
            // 调用服务层方法进行对齐操作
            return alignmentService.alignTextAndImage();
        } catch (Exception e) {
            e.printStackTrace();
            // 可以返回错误信息或其他处理
            return null;
        }
    }

    @GetMapping("/listResults")
    public List<AlignmentVO> listAll() {
        return alignmentService.getAllResults();
    }

    @GetMapping("/listPatients")
    public List<String> listPatients() {
        return alignmentService.getPatients();
    }
}
