package DataAlignmentAndFusionApplication.controller;


import DataAlignmentAndFusionApplication.model.vo.HomeOverviewVO;
import DataAlignmentAndFusionApplication.service.HomeAggregateService;
import DataAlignmentAndFusionApplication.util.Result;
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

}
