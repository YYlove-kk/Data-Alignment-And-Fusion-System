package DataAlignmentAndFusionApplication.service;

import DataAlignmentAndFusionApplication.model.entity.ReduceRecord;

import java.util.List;

public interface DimReduceService {

    void reduce();

    List<ReduceRecord> getRecords();

    List<String> getSourceIds();
}
