package DataAlignmentAndFusionApplication.controller;

import DataAlignmentAndFusionApplication.algorithm.data.GraphNode;
import DataAlignmentAndFusionApplication.model.dto.ConstructionDTO;
import DataAlignmentAndFusionApplication.model.vo.GraphVO;
import DataAlignmentAndFusionApplication.service.FusionService;
import DataAlignmentAndFusionApplication.util.Result;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
@RequestMapping("/api/fusion")
public class FusionController {

//    private final FusionService fusionService;


    @PostMapping("/build")
    public Result<GraphVO> buildGraph(@RequestBody ConstructionDTO dto) {
//        return Result.success(fusionService.buildKnowledgeGraph(dto));
        return null;
    }

    @PostMapping("/nodes/batch")
    public Result<Void> saveNodes(@RequestBody List<GraphNode> nodes) {
//        fusionService.batchSaveNodes(nodes);
//        return Result.success();
        return null;
    }
}