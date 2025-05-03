package DataAlignmentAndFusionApplication.mapper;

import DataAlignmentAndFusionApplication.model.entity.EmbedRecord;
import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import org.apache.ibatis.annotations.Param;

import java.util.List;

/**
* @author 29857
* @description 针对表【embed_record】的数据库操作Mapper
* @createDate 2025-05-04 00:51:53
* @Entity DataAlignmentAndFusionApplication.model.entity.EmbedRecord
*/
public interface EmbedRecordMapper extends BaseMapper<EmbedRecord> {
    List<String> selectDistinctPatientIdsBySourceId(@Param("sourceId") String sourceId);
}




