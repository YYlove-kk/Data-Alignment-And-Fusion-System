package DataAlignmentAndFusionApplication.service;

import DataAlignmentAndFusionApplication.model.dto.GraphReq;
import DataAlignmentAndFusionApplication.model.vo.GraphVO;

public interface GraphService {
    GraphVO buildKnowledgeGraph(GraphReq req);
}