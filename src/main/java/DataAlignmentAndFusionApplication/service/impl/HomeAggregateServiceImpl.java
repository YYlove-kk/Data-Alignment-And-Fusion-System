package DataAlignmentAndFusionApplication.service.impl;

import DataAlignmentAndFusionApplication.model.vo.HomeOverviewVO;
import DataAlignmentAndFusionApplication.service.AlignmentService;
import DataAlignmentAndFusionApplication.service.AnalysisService;
import DataAlignmentAndFusionApplication.service.HomeAggregateService;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

@Service
@RequiredArgsConstructor
public class HomeAggregateServiceImpl implements HomeAggregateService {
    private final UploadService uploadService;
    private final AlignmentService alignmentService;
    //    private final FusionService fusionService;
    private final AnalysisService analysisService;
//    private final GraphVisualService visualService;

    @Override
    public HomeOverviewVO getHomeOverview() {
        HomeOverviewVO vo = new HomeOverviewVO();

//        // 1. 获取各模块统计
//        vo.setUploadStats(uploadService.getUploadStats());
//        vo.setAlignmentStats(alignmentService.getAlignmentStats());
//        vo.setGraphStats(fusionService.getGraphStats());
//        vo.setAnalysisStats(analysisService.getAnalysisStats());
//
//        // 2. 生成简化图谱
//        vo.setGraphPreview(
//                visualService.generatePreviewGraph(
//                        fusionService.getRecentGraphData()
//                )
//        );
//
//        // 3. 合并最近活动
//        vo.setActivities(mergeRecentActivities(
//                uploadService.getRecentActivities(),
//                alignmentService.getRecentActivities()
//        ));
//
        return vo;
    }
//
//        @Override
//        public GlobalSearchResultVO globalSearch (GlobalSearchDTO dto){
//            // 根据scope选择搜索模块
//            return switch (dto.getScope()) {
//                case UPLOAD -> uploadService.search(dto);
//                case ALIGNMENT -> alignmentService.search(dto);
//                case GRAPH -> fusionService.searchGraph(dto);
//                case ANALYSIS -> analysisService.search(dto);
//                default -> crossModuleSearch(dto); // 跨模块联合搜索
//            };
//        }
    }
