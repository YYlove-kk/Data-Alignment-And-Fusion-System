package DataAlignmentAndFusionApplication.model.vo;

import lombok.Data;

import java.util.List;

@Data
public class PageVO<T> {
    private List<T> list;
    private Long total;
    private Integer pageSize;
    private Integer currentPage;
}
