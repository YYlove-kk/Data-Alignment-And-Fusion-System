package DataAlignmentAndFusionApplication.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import java.util.Date;
import lombok.Data;

/**
 * 
 * @TableName upload_record
 */
@TableName(value ="upload_record")
@Data
public class UploadRecord {
    /**
     * 
     */
    @TableId(type = IdType.AUTO)
    private Long id;

    /**
     * 
     */
    private String dataSourceName;

    /**
     * 
     */
    private String modalityType;

    /**
     * 
     */
    private String institution;

    /**
     * 
     */
    private String filePath;

    /**
     * 
     */
    private String fileName;

    /**
     * 
     */
    private Long fileSize;

    /**
     * 
     */
    private String status;

    /**
     * 
     */
    private String errorDetail;

    /**
     * 
     */
    private Date uploadTime;

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
        UploadRecord other = (UploadRecord) that;
        return (this.getId() == null ? other.getId() == null : this.getId().equals(other.getId()))
            && (this.getDataSourceName() == null ? other.getDataSourceName() == null : this.getDataSourceName().equals(other.getDataSourceName()))
            && (this.getModalityType() == null ? other.getModalityType() == null : this.getModalityType().equals(other.getModalityType()))
            && (this.getInstitution() == null ? other.getInstitution() == null : this.getInstitution().equals(other.getInstitution()))
            && (this.getFilePath() == null ? other.getFilePath() == null : this.getFilePath().equals(other.getFilePath()))
            && (this.getFileName() == null ? other.getFileName() == null : this.getFileName().equals(other.getFileName()))
            && (this.getFileSize() == null ? other.getFileSize() == null : this.getFileSize().equals(other.getFileSize()))
            && (this.getStatus() == null ? other.getStatus() == null : this.getStatus().equals(other.getStatus()))
            && (this.getErrorDetail() == null ? other.getErrorDetail() == null : this.getErrorDetail().equals(other.getErrorDetail()))
            && (this.getUploadTime() == null ? other.getUploadTime() == null : this.getUploadTime().equals(other.getUploadTime()));
    }

    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        result = prime * result + ((getId() == null) ? 0 : getId().hashCode());
        result = prime * result + ((getDataSourceName() == null) ? 0 : getDataSourceName().hashCode());
        result = prime * result + ((getModalityType() == null) ? 0 : getModalityType().hashCode());
        result = prime * result + ((getInstitution() == null) ? 0 : getInstitution().hashCode());
        result = prime * result + ((getFilePath() == null) ? 0 : getFilePath().hashCode());
        result = prime * result + ((getFileName() == null) ? 0 : getFileName().hashCode());
        result = prime * result + ((getFileSize() == null) ? 0 : getFileSize().hashCode());
        result = prime * result + ((getStatus() == null) ? 0 : getStatus().hashCode());
        result = prime * result + ((getErrorDetail() == null) ? 0 : getErrorDetail().hashCode());
        result = prime * result + ((getUploadTime() == null) ? 0 : getUploadTime().hashCode());
        return result;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(getClass().getSimpleName());
        sb.append(" [");
        sb.append("Hash = ").append(hashCode());
        sb.append(", id=").append(id);
        sb.append(", dataSourceName=").append(dataSourceName);
        sb.append(", modalityType=").append(modalityType);
        sb.append(", institution=").append(institution);
        sb.append(", filePath=").append(filePath);
        sb.append(", fileName=").append(fileName);
        sb.append(", fileSize=").append(fileSize);
        sb.append(", status=").append(status);
        sb.append(", errorDetail=").append(errorDetail);
        sb.append(", uploadTime=").append(uploadTime);
        sb.append("]");
        return sb.toString();
    }
}