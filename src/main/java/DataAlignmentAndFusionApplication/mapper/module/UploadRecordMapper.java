package DataAlignmentAndFusionApplication.mapper.module;

import DataAlignmentAndFusionApplication.model.entity.UploadRecord;
import org.apache.ibatis.annotations.Mapper;  // 导入注解
import com.baomidou.mybatisplus.core.mapper.BaseMapper;


@Mapper
public interface UploadRecordMapper extends BaseMapper<UploadRecord> {

}