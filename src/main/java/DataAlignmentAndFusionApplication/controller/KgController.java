package DataAlignmentAndFusionApplication.controller;


import DataAlignmentAndFusionApplication.model.DTO.ConstructionDTO;
import DataAlignmentAndFusionApplication.model.DTO.FusionDTO;
import DataAlignmentAndFusionApplication.model.vo.*;
import DataAlignmentAndFusionApplication.service.KgService;
import DataAlignmentAndFusionApplication.util.Result;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotNull;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;


@RestController
@RequestMapping("/kg")
@Validated
public class KgController {

    private final KgService kgService;

    public KgController(KgService kgService) {
        this.kgService = kgService;
    }

    //创建
    @PostMapping("/build")
    public Result<GraphVO> buildKnowledgeGraph(
            @RequestBody @Valid ConstructionDTO dto,
            @RequestHeader("X-User-Id") @NotNull Long userId) {
        return Result.success(kgService.buildKnowledgeGraph(dto));
    }

    //融合
    @PostMapping("/fuse")
    public Result<FusionVO> fuseGraphs(
            @RequestBody @Valid FusionDTO dto) {
        return Result.success(kgService.fuseGraphs(dto));
    }

    //可视化
    @GetMapping("/visualization/{graphId}")
    public Result<GraphVisualizationVO> getVisualization(
            @PathVariable Long graphId) {
        return Result.success(kgService.getGraphVisualization(graphId));
    }

    //节点详情
    @GetMapping("/node/{nodeId}")
    public Result<NodeDetailVO> getNodeDetails(
            @PathVariable String nodeId,
            @RequestParam Long graphId) {
        return Result.success(kgService.getNodeDetails(nodeId, graphId));
    }

    //历史记录
    @GetMapping("/history")
    public Result<PageVO<ConstructionRecordVO>> getConstructionHistory(
            @RequestHeader("X-User-Id") @NotNull Long userId,
            @RequestParam(defaultValue = "1") Integer page,
            @RequestParam(defaultValue = "10") Integer size) {
        return Result.success(kgService.getConstructionHistory(userId, page, size));
    }
}
