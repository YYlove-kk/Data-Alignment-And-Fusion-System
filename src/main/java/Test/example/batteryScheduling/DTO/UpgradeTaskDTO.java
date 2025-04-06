package Test.example.batteryScheduling.DTO;

import com.baomidou.mybatisplus.annotation.TableId;
import com.fasterxml.jackson.annotation.JsonFormat;
import org.springframework.format.annotation.DateTimeFormat;

import java.util.Date;

public class UpgradeTaskDTO {
    /**
     * 主键id
     */
    @TableId
    private String applicationNumber;
    /*
        申请人名称
     */
    private String applicationName;//

    /*
        申请主题
     */
    private String applicationSubject;

    /*
        申请人所属部门
     */
    private String applicationDepartment;

    /*
         申请原因
     */
    private String applicationReason;

    /*
         升级包名称
     */
    private String upgrade_package_name;
    /*
        申请状态
     */
    private String applicationStatus;



    /*
                申请类版本
             */
    private String applicationVersion;


    /*
            申请类型
         */
    private String applicationType;

    /*
        申请请求时间
     */
    @JsonFormat(locale="zh", timezone="GMT+8", pattern="yyyy-MM-dd HH:mm:ss")
    @DateTimeFormat
    private Date applicationTimes;

    public String getApplicationNumber() {
        return applicationNumber;
    }

    public void setApplicationNumber(String applicationNumber) {
        this.applicationNumber = applicationNumber;
    }

    public String getApplicationName() {
        return applicationName;
    }

    public void setApplicationName(String applicationName) {
        this.applicationName = applicationName;
    }

    public String getApplicationSubject() {
        return applicationSubject;
    }

    public void setApplicationSubject(String applicationSubject) {
        this.applicationSubject = applicationSubject;
    }

    public String getApplicationDepartment() {
        return applicationDepartment;
    }

    public void setApplicationDepartment(String applicationDepartment) {
        this.applicationDepartment = applicationDepartment;
    }

    public String getApplicationReason() {
        return applicationReason;
    }

    public void setApplicationReason(String applicationReason) {
        this.applicationReason = applicationReason;
    }

    public String getUpgrade_package_name() {
        return upgrade_package_name;
    }

    public void setUpgrade_package_name(String upgrade_package_name) {
        this.upgrade_package_name = upgrade_package_name;
    }

    public String getApplicationStatus() {
        return applicationStatus;
    }

    public void setApplicationStatus(String applicationStatus) {
        this.applicationStatus = applicationStatus;
    }

    public Date getApplicationTimes() {
        return applicationTimes;
    }

    public void setApplicationTimes(Date applicationTimes) {
        this.applicationTimes = applicationTimes;
    }
    public String getApplicationType() {
        return applicationType;
    }

    public void setApplicationType(String applicationType) {
        this.applicationType = applicationType;
    }

    public String getApplicationVersion() {
        return applicationVersion;
    }

    public void setApplicationVersion(String applicationVersion) {
        this.applicationVersion = applicationVersion;
    }
}
