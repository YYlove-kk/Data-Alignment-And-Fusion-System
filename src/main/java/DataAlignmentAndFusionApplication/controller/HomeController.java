package DataAlignmentAndFusionApplication.controller;

import DataAlignmentAndFusionApplication.model.dto.GlobalSearchDTO;
import DataAlignmentAndFusionApplication.model.vo.HomeOverviewVO;
import DataAlignmentAndFusionApplication.service.HomeAggregateService;
import DataAlignmentAndFusionApplication.util.Result;
import jakarta.validation.Valid;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/home")
public class HomeController {

    private HomeAggregateService homeService;

    // 获取首页聚合数据
    @GetMapping("/overview")
    public Result<HomeOverviewVO> getOverview() {
        return Result.success(homeService.getHomeOverview());
    }

    // 全局搜索
//    @PostMapping("/global-search")
//    public Result<GlobalSearchResultVO> globalSearch(
//            @RequestBody @Valid GlobalSearchDTO dto) {
//        return Result.success(homeService.globalSearch(dto));
//    }
}
