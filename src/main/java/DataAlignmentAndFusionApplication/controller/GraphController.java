package DataAlignmentAndFusionApplication.controller;

import DataAlignmentAndFusionApplication.model.dto.GraphReq;
import DataAlignmentAndFusionApplication.model.vo.GraphVO;
import DataAlignmentAndFusionApplication.service.FusionRecordService;
import DataAlignmentAndFusionApplication.service.GraphService;
import DataAlignmentAndFusionApplication.service.KpcaService;
import DataAlignmentAndFusionApplication.util.Result;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/graph")
@CrossOrigin
public class GraphController {

    @Autowired
    private GraphService graphService;

    @Autowired
    private FusionRecordService fusionRecordService;

    @Autowired
    private KpcaService kpcaService;

    @PostMapping("/reduce")
    public Result<String> reduceVectors(GraphReq req) {
        return kpcaService.runKpcaReduction(req);
    }

    @GetMapping("/build")
    public GraphVO buildKnowledgeGraph(GraphReq req) {
        return graphService.buildKnowledgeGraph(req);
    }

    @PostMapping("/fuse")
    public GraphVO fuseKnowledgeGraph(GraphReq req) {
        return fusionRecordService.fuseGraph(req);
    }

    @GetMapping("/availableGraph")
    public List<Integer> getAvailableGraph() {
        return fusionRecordService.getAvailableGraph();
    }
}
