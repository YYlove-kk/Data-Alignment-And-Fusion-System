package Test.example.batteryScheduling.domain;

import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import com.fasterxml.jackson.annotation.JsonFormat;
import lombok.Data;
import lombok.Getter;
import org.springframework.beans.BeanUtils;
import org.springframework.format.annotation.DateTimeFormat;

import java.lang.reflect.InvocationTargetException;
import java.util.Date;

@Data
@TableName(value = "BatteryModule", autoResultMap = true)
public class BatteryModule {

    /**
     * 主键id
     */
	@Getter
    @TableId
    private Long id;

    /*
        零件号
     */

    @Getter
    private String partNumber;

    /*
        托号
     */
    @Getter
    private String setNumber;//托号

    /*
        MAC地址
     */
    @Getter
    private String macAddress;

    /*
        模组名称
         */
    @Getter
    private String moduleName;

    /*
    模组SOC
     */
    @Getter
    private double moduleSoc;

    /*
    模组温度
     */
    @Getter
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
    @Getter
    private String scriptVersion;

    /*
    更新时间
     */
    @JsonFormat(locale="zh", timezone="GMT+8", pattern="yyyy-MM-dd HH:mm:ss")
    @DateTimeFormat
    private Date add_time;

    /*
    入库状态，未入库为0，正常为1
     */
    private String entryStatus;

    @Getter
    private String module_divide;

    /*
    跟踪码
     */
    private String traceCode;


    public void setId(Long id) {
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

    public void setModuleSoc(double moduleSoc) {
        this.moduleSoc = moduleSoc;
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

    public void setScriptVersion(String scriptVersion) {
        this.scriptVersion = scriptVersion;
    }

    public Date getTime() {
        return add_time;
    }

    public void setTime(Date add_time) {
        this.add_time = add_time;
    }

    public void setEntryStatus(String entryStatus) {
        this.entryStatus = entryStatus;
    }

    public void setModuleName(String moduleName) {
        this.moduleName = moduleName;
    }

    public void setModuleVolt(double moduleVolt) {
        this.moduleVolt = moduleVolt;
    }
    public void setModule_divide(String module_divide) {
        this.module_divide = module_divide;
    }
    public String getModule_divide() {
        return module_divide;
    }

    public void setWilVersion(String wilVersion) {
        this.wilVersion = wilVersion;
    }

    public void setAdd_time(Date add_time) {
        this.add_time = add_time;
    }

    public void setTraceCode(String traceCode) {
        this.traceCode = traceCode;
    }
}
