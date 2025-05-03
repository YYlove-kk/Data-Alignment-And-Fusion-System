package DataAlignmentAndFusionApplication.service;


import DataAlignmentAndFusionApplication.model.vo.HomeOverviewVO;
import jakarta.validation.Valid;

public interface HomeAggregateService {
    HomeOverviewVO getHomeOverview();

}
