package DataAlignmentAndFusionApplication.service;

import DataAlignmentAndFusionApplication.model.dto.GraphReq;
import DataAlignmentAndFusionApplication.model.entity.FusionRecord;
import DataAlignmentAndFusionApplication.model.vo.GraphVO;
import com.baomidou.mybatisplus.extension.service.IService;

import java.util.List;

/**
* @author 29857
* @description 针对表【fusion_record】的数据库操作Service
* @createDate 2025-04-30 21:22:01
*/
public interface FusionRecordService extends IService<FusionRecord> {

    GraphVO fuseGraph(GraphReq req,String modeName);

    List<Integer> getAvailableGraph();
}
