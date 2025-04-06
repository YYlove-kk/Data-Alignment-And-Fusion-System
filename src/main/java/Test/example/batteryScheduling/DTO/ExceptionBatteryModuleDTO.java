package Test.example.batteryScheduling.DTO;

import com.fasterxml.jackson.annotation.JsonFormat;
import org.springframework.format.annotation.DateTimeFormat;

import java.util.Date;

public class ExceptionBatteryModuleDTO {


    private String id;

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


    @JsonFormat(locale="zh", timezone="GMT+8", pattern="yyyy-MM-dd HH:mm:ss")
    @DateTimeFormat
    private Date add_time;

    private String exceptionStatus;


    public String getId() {
        return id;
    }

    public void setId(String id) {
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

    public String getModuleName() {
        return moduleName;
    }

    public void setModuleName(String moduleName) {
        this.moduleName = moduleName;
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

    public String getWilVersion() {
        return wilVersion;
    }

    public void setWilVersion(String wilVersion) {
        this.wilVersion = wilVersion;
    }

    public String getScriptVersion() {
        return scriptVersion;
    }

    public void setScriptVersion(String scriptVersion) {
        this.scriptVersion = scriptVersion;
    }

    public String getExceptionStatus() {
        return exceptionStatus;
    }

    public void setExceptionStatus(String exceptionStatus) {
        this.exceptionStatus = exceptionStatus;
    }


    public Date getAdd_time() {
        return add_time;
    }

    public void setAdd_time(Date add_time) {
        this.add_time = add_time;
    }
}
