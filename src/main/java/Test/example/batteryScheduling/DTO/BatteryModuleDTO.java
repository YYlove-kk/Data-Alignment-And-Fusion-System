package Test.example.batteryScheduling.DTO;

import com.fasterxml.jackson.annotation.JsonFormat;
import lombok.Getter;
import org.springframework.format.annotation.DateTimeFormat;

import java.util.Date;

@Getter
public class BatteryModuleDTO {


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
            模组电压
             */
    private double moduleVolt;


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

    private String module_divide;

    /*
   入库状态，未入库为0，正常为1
    */
    private String entryStatus;

    /*
    跟踪码
     */
    private String traceCode;


    public void setId(String id) {
        this.id = id;
    }

    public void setPartNumber(String partNumber) {
        this.partNumber = partNumber;
    }

    public void setSetNumber(String setNumber) {
        this.setNumber = setNumber;
    }

    public void setMacAddress(String macAddress) {
        this.macAddress = macAddress;
    }

    public void setModuleName(String moduleName) {
        this.moduleName = moduleName;
    }

    public void setModuleSoc(double moduleSoc) {
        this.moduleSoc = moduleSoc;
    }

    public void setModuleTemperature(double moduleTemperature) {
        this.moduleTemperature = moduleTemperature;
    }

    public void setWilVersion(String wilVersion) {
        this.wilVersion = wilVersion;
    }

    public void setScriptVersion(String scriptVersion) {
        this.scriptVersion = scriptVersion;
    }


    public void setAdd_time(Date add_time) {
        this.add_time = add_time;
    }

    public void setModuleVolt(double moduleVolt) {
        this.moduleVolt = moduleVolt;
    }

    public void setModule_divide(String module_divide) {
        this.module_divide = module_divide;
    }

    public void setEntryStatus(String entryStatus) {
        this.entryStatus = entryStatus;
    }

    public void setTraceCode(String traceCode) {
        this.traceCode = traceCode;
    }
}
