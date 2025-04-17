package DataAlignmentAndFusionApplication.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import java.util.Date;
import lombok.Data;

/**
 * 
 * @TableName alignment_config
 */
@TableName(value ="alignment_config")
@Data
public class AlignmentConfig {
    /**
     * 
     */
    @TableId(type = IdType.AUTO)
    private Long id;

    /**
     * 用户ID
     */
    private String userId;

    /**
     * 时序对齐算法
     */
    private String timeSeriesAlgo;

    /**
     * 时间粒度
     */
    private String timeGranularity;

    /**
     * 语义模型
     */
    private String semanticModel;

    /**
     * 相似度阈值
     */
    private Double similarityThreshold;

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
        AlignmentConfig other = (AlignmentConfig) that;
        return (this.getId() == null ? other.getId() == null : this.getId().equals(other.getId()))
            && (this.getUserId() == null ? other.getUserId() == null : this.getUserId().equals(other.getUserId()))
            && (this.getTimeSeriesAlgo() == null ? other.getTimeSeriesAlgo() == null : this.getTimeSeriesAlgo().equals(other.getTimeSeriesAlgo()))
            && (this.getTimeGranularity() == null ? other.getTimeGranularity() == null : this.getTimeGranularity().equals(other.getTimeGranularity()))
            && (this.getSemanticModel() == null ? other.getSemanticModel() == null : this.getSemanticModel().equals(other.getSemanticModel()))
            && (this.getSimilarityThreshold() == null ? other.getSimilarityThreshold() == null : this.getSimilarityThreshold().equals(other.getSimilarityThreshold()))
            && (this.getCreateTime() == null ? other.getCreateTime() == null : this.getCreateTime().equals(other.getCreateTime()));
    }

    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        result = prime * result + ((getId() == null) ? 0 : getId().hashCode());
        result = prime * result + ((getUserId() == null) ? 0 : getUserId().hashCode());
        result = prime * result + ((getTimeSeriesAlgo() == null) ? 0 : getTimeSeriesAlgo().hashCode());
        result = prime * result + ((getTimeGranularity() == null) ? 0 : getTimeGranularity().hashCode());
        result = prime * result + ((getSemanticModel() == null) ? 0 : getSemanticModel().hashCode());
        result = prime * result + ((getSimilarityThreshold() == null) ? 0 : getSimilarityThreshold().hashCode());
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
        sb.append(", userId=").append(userId);
        sb.append(", timeSeriesAlgo=").append(timeSeriesAlgo);
        sb.append(", timeGranularity=").append(timeGranularity);
        sb.append(", semanticModel=").append(semanticModel);
        sb.append(", similarityThreshold=").append(similarityThreshold);
        sb.append(", createTime=").append(createTime);
        sb.append("]");
        return sb.toString();
    }
}