package Test.example.batteryScheduling.domain;

import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import com.fasterxml.jackson.annotation.JsonFormat;
import lombok.Data;
import org.springframework.format.annotation.DateTimeFormat;

import java.util.Date;

@Data
@TableName(value = "ExceptionBatteryModule", autoResultMap = true)
public class ExceptionBatteryModule {

    /**
     * 主键id
     */
	@TableId
    private Long id;

    /*
        零件号
     */

    private String partNumber;

    /*
        托号
     */
    private String setNumber;//托号

    /*
        MAC地址
     */
    private String macAddress;

    /*
        模组名称
         */
    private String moduleName;

    /*
    模组SOC
     */
    private double moduleSoc;

    /*
    模组温度
     */
    private double moduleTemperature;

	/*
	模组版本
	 */

    private String wilVersion;


    /*
        脚本版本
         */
    private String scriptVersion;

    /*
    更新时间
     */
    @JsonFormat(locale="zh", timezone="GMT+8", pattern="yyyy-MM-dd HH:mm:ss")
    @DateTimeFormat
    private Date add_time;

    /*
    异常状态，SOC异常---1，温度异常---2，自放电异常---3，通讯异常---4，性能异常---5
     */
    private String exceptionStatus;


    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getPartNumber() {
        return partNumber;
    }

    public void setPartNumber(String partNumber) {
        this.partNumber = partNumber;
    }

    public String getSetNumber() {
        return setNumber;
    }

    public void setSetNumber(String setNumber) {
        this.setNumber = setNumber;
    }

    public String getMacAddress() {
        return macAddress;
    }

    public void setMacAddress(String macAddress) {
        this.macAddress = macAddress;
    }

    public double getModuleSoc() {
        return moduleSoc;
    }

    public void setModuleSoc(double moduleSoc) {
        this.moduleSoc = moduleSoc;
    }

    public double getModuleTemperature() {
        return moduleTemperature;
    }

    public void setModuleTemperature(double moduleTemperature) {
        this.moduleTemperature = moduleTemperature;
    }

    public String getModuleVersion() {
        return wilVersion;
    }

    public void setModuleVersion(String wilVersion) {
        this.wilVersion = wilVersion;
    }

    public String getScriptVersion() {
        return scriptVersion;
    }

    public void setScriptVersion(String scriptVersion) {
        this.scriptVersion = scriptVersion;
    }

    public Date getTime() {
        return add_time;
    }

    public void setTime(Date add_time) {
        this.add_time = add_time;
    }

    public String getExceptionEntryStatus() {
        return exceptionStatus;
    }

    public void setExceptionEntryStatus(String exceptionStatus) {
        this.exceptionStatus = exceptionStatus;
    }

    public String getModuleName() {
        return moduleName;
    }

    public void setModuleName(String moduleName) {
        this.moduleName = moduleName;
    }
}
