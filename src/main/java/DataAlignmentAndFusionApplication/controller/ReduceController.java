package DataAlignmentAndFusionApplication.controller;

import DataAlignmentAndFusionApplication.model.entity.ReduceRecord;
import DataAlignmentAndFusionApplication.service.DimReduceService;
import DataAlignmentAndFusionApplication.util.Result;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
@RequestMapping("/api/dimReduce")
public class ReduceController {

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

    @GetMapping("/records")
    public List<ReduceRecord> records() {
        return dimReduceService.getRecords();
    }

    @GetMapping("/sourceIds" )
    public List<String> getSourceIds() {
        return dimReduceService.getSourceIds();
    }
}