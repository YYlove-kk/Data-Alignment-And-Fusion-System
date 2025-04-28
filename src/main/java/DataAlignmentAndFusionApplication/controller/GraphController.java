package DataAlignmentAndFusionApplication.controller;

import DataAlignmentAndFusionApplication.model.vo.GraphVO;
import DataAlignmentAndFusionApplication.service.GraphService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/graph")
@CrossOrigin
public class GraphController {

    @Autowired
    private GraphService graphService;

    @GetMapping("/build")
    public GraphVO buildKnowledgeGraph() {
        return graphService.buildKnowledgeGraph();
    }
}
