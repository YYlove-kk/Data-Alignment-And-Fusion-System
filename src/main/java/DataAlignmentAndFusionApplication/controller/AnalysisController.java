package DataAlignmentAndFusionApplication.controller;

import DataAlignmentAndFusionApplication.model.dto.AnalysisDTO;
import DataAlignmentAndFusionApplication.model.entity.PatientAnalysisRecord;
import DataAlignmentAndFusionApplication.model.vo.AnalysisHistoryVO;
import DataAlignmentAndFusionApplication.model.vo.AnalysisResultVO;
import DataAlignmentAndFusionApplication.model.vo.PageVO;
import DataAlignmentAndFusionApplication.service.AnalysisService;
import DataAlignmentAndFusionApplication.util.Result;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import jakarta.validation.Valid;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/analysis")
public class AnalysisController {

    public AnalysisService analysisService;


    @PostMapping("/perform")
    public Result<AnalysisResultVO> performAnalysis(
            @RequestBody @Valid AnalysisDTO dto,
            @RequestHeader("X-User-Id") Long userId) {

        dto.setUserId(userId);
        return Result.success(analysisService.performAnalysis(dto));
    }

    @GetMapping("/history")
    public Result<PageVO<AnalysisHistoryVO>> getHistory(
            @RequestHeader("X-User-Id") Long userId,
            @RequestParam(defaultValue = "1") Integer page,
            @RequestParam(defaultValue = "10") Integer size) {

        Page<PatientAnalysisRecord> records = analysisService.getHistory(userId, page, size);
        return null;
    }
}