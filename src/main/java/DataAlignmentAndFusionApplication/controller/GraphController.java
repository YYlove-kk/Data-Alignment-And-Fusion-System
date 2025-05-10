package DataAlignmentAndFusionApplication.controller;

import DataAlignmentAndFusionApplication.model.dto.DeleteReq;
import DataAlignmentAndFusionApplication.model.dto.GraphReq;
import DataAlignmentAndFusionApplication.model.vo.GraphVO;
import DataAlignmentAndFusionApplication.service.FusionRecordService;
import DataAlignmentAndFusionApplication.service.GraphService;
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


    @PostMapping("/build")
    public GraphVO buildKnowledgeGraph(@RequestBody GraphReq req) {
        return graphService.buildKnowledgeGraph(req);
    }

    @PostMapping("/fuse")
    public GraphVO fuseKnowledgeGraph(@RequestBody GraphReq req,@RequestParam  String modeName) {
        return fusionRecordService.fuseGraph(req, modeName);
    }

    @GetMapping("/availableGraph")
    public List<Integer> getAvailableGraph() {
        return fusionRecordService.getAvailableGraph();
    }

    @PostMapping("/deleteEdge")
    public Result<String> deleteEdge(@RequestBody DeleteReq req) {
        return graphService.deleteEdge(req);
    }

    @GetMapping("/home")
    public GraphVO queryGraph(@RequestParam(value = "tag", required = false) Integer tag) {
        return graphService.getGraph(tag);
    }
}
