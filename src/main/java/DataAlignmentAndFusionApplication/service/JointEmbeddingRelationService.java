package DataAlignmentAndFusionApplication.service;

import DataAlignmentAndFusionApplication.model.entity.JointEmbeddingRelation;
import com.baomidou.mybatisplus.extension.service.IService;

import java.util.List;

/**
* @author 29857
* @description 针对表【joint_embedding_relation】的数据库操作Service
* @createDate 2025-05-03 23:52:39
*/
public interface JointEmbeddingRelationService extends IService<JointEmbeddingRelation> {
    void saveBatch(List<JointEmbeddingRelation> relations);

}
