package DataAlignmentAndFusionApplication.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import java.util.Date;
import lombok.Data;

/**
 * 
 * @TableName alignment_result
 */
@TableName(value ="alignment_result")
@Data
public class AlignmentResult {
    /**
     * 
     */
    @TableId(type = IdType.AUTO)
    private Long id;

    /**
     * 关联的配置ID
     */
    private Long configId;

    /**
     * 对齐结果JSON
     */
    private String resultJson;

    /**
     * 图表数据JSON
     */
    private String chartData;

    /**
     * 平均相似度
     */
    private Double avgSimilarity;

    /**
     * 任务状态
     */
    private String status;

    /**
     * 
     */
    private Date createTime;

    @Override
    public boolean equals(Object that) {
        if (this == that) {
            return true;
        }
        if (that == null) {
            return false;
        }
        if (getClass() != that.getClass()) {
            return false;
        }
        AlignmentResult other = (AlignmentResult) that;
        return (this.getId() == null ? other.getId() == null : this.getId().equals(other.getId()))
            && (this.getConfigId() == null ? other.getConfigId() == null : this.getConfigId().equals(other.getConfigId()))
            && (this.getResultJson() == null ? other.getResultJson() == null : this.getResultJson().equals(other.getResultJson()))
            && (this.getChartData() == null ? other.getChartData() == null : this.getChartData().equals(other.getChartData()))
            && (this.getAvgSimilarity() == null ? other.getAvgSimilarity() == null : this.getAvgSimilarity().equals(other.getAvgSimilarity()))
            && (this.getStatus() == null ? other.getStatus() == null : this.getStatus().equals(other.getStatus()))
            && (this.getCreateTime() == null ? other.getCreateTime() == null : this.getCreateTime().equals(other.getCreateTime()));
    }

    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        result = prime * result + ((getId() == null) ? 0 : getId().hashCode());
        result = prime * result + ((getConfigId() == null) ? 0 : getConfigId().hashCode());
        result = prime * result + ((getResultJson() == null) ? 0 : getResultJson().hashCode());
        result = prime * result + ((getChartData() == null) ? 0 : getChartData().hashCode());
        result = prime * result + ((getAvgSimilarity() == null) ? 0 : getAvgSimilarity().hashCode());
        result = prime * result + ((getStatus() == null) ? 0 : getStatus().hashCode());
        result = prime * result + ((getCreateTime() == null) ? 0 : getCreateTime().hashCode());
        return result;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(getClass().getSimpleName());
        sb.append(" [");
        sb.append("Hash = ").append(hashCode());
        sb.append(", id=").append(id);
        sb.append(", configId=").append(configId);
        sb.append(", resultJson=").append(resultJson);
        sb.append(", chartData=").append(chartData);
        sb.append(", avgSimilarity=").append(avgSimilarity);
        sb.append(", status=").append(status);
        sb.append(", createTime=").append(createTime);
        sb.append("]");
        return sb.toString();
    }
}