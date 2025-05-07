package DataAlignmentAndFusionApplication.service;

import DataAlignmentAndFusionApplication.model.dto.DeleteReq;
import DataAlignmentAndFusionApplication.model.dto.GraphReq;
import DataAlignmentAndFusionApplication.model.vo.GraphVO;
import DataAlignmentAndFusionApplication.util.Result;

public interface GraphService {
    GraphVO buildKnowledgeGraph(GraphReq req);
    Result<String> deleteEdge(DeleteReq req);
}