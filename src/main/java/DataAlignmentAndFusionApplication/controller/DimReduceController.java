package DataAlignmentAndFusionApplication.controller;

import DataAlignmentAndFusionApplication.service.DimReduceService;
import DataAlignmentAndFusionApplication.util.Result;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/dimReduce")
public class DimReduceController {

    @Autowired
    private DimReduceService dimReduceService;

    @PostMapping("/reduce")
    public Result<String> dimReduce() {
        try {
            // 调用服务进行降维处理
            dimReduceService.reduce();
            return Result.success("DimReduction success.");
        } catch (Exception e) {
            return Result.error(500, e.getMessage());
        }
    }
}