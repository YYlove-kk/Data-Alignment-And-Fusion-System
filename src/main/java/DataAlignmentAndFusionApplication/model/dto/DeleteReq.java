package DataAlignmentAndFusionApplication.model.dto;

import lombok.Data;

@Data
public class DeleteReq {
    private String uuid1;      // 第一个节点 UUID
    private String uuid2;      // 第二个节点 UUID
    private Integer tag;       // 边的 tag 属性
    private String edgeType;   // 边的类型，例如 "MULTI_MODAL_SIMILAR"
}
