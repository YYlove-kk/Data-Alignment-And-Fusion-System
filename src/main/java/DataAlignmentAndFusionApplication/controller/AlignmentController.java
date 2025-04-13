package DataAlignmentAndFusionApplication.controller;

import DataAlignmentAndFusionApplication.model.DTO.AlignmentConfigDTO;
import DataAlignmentAndFusionApplication.model.DTO.AlignmentExecuteDTO;
import DataAlignmentAndFusionApplication.model.vo.AlignmentResultVO;
import DataAlignmentAndFusionApplication.model.vo.PageVO;
import DataAlignmentAndFusionApplication.service.AlignmentService;
import DataAlignmentAndFusionApplication.util.Result;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/alignment")
@RequiredArgsConstructor
public class AlignmentController {
    private final AlignmentService alignmentService;

    // 保存对齐配置
    @PostMapping("/config/save")
    public Result<Void> saveConfig(
            @Valid @RequestBody AlignmentConfigDTO dto,
            @RequestHeader("X-User-Id") Long userId
    ) {
        alignmentService.saveConfig(dto, userId);
        return null;
    }

    // 执行对齐操作
    @PostMapping("/execute")
    public Result<Long> execute(
            @Valid @RequestBody AlignmentExecuteDTO dto
    ) {
        return alignmentService.executeAlignment(dto);
    }

    // 获取对齐结果
    @GetMapping("/result/{id}")
    public Result<AlignmentResultVO> getResult(@PathVariable Long id) {
        return alignmentService.getResultById(id);
    }

    // 分页查询历史记录
    @GetMapping("/history")
    public Result<PageVO<AlignmentResultVO>> listHistory(
            @RequestParam Long userId,
            @RequestParam(defaultValue = "1") Integer page,
            @RequestParam(defaultValue = "10") Integer size
    ) {
        return alignmentService.listHistory(userId, page, size);
    }
}
