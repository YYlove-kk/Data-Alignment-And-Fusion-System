package Test.example.batteryScheduling.domain;

import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import com.fasterxml.jackson.annotation.JsonFormat;
import lombok.Data;
import org.springframework.format.annotation.DateTimeFormat;

import java.util.Date;

@Data
@TableName(value = "device", autoResultMap = true)
public class Device {

    /**
     * 主键id
     */
	@TableId
    private Long id;
    /*
        设备名称
     */
    private String deviceName;//

    /*
        设备监控模组数量
         */
    private int deviceModuleAccount;

    /*
        设备启动日期，0表示周一
         */
    private int deviceStartDay;

    /*
        设备启动日期的具体小时
         */
    private int deviceStartHour;

    /*
        设备启动日期的具体分钟数
         */
    private int deviceStartMinute;

    /*
        设备状态
         */
    private String deviceStatus;

    /*
        设备状态更新时间
     */
    @JsonFormat(locale="zh", timezone="GMT+8", pattern="yyyy-MM-dd HH:mm:ss")
    @DateTimeFormat
    private Date deviceAddtime;

    public void setId(Long id) {
        this.id = id;
    }

    public void setDeviceName(String deviceName) {
        this.deviceName = deviceName;
    }

    public void setDeviceStartDay(int deviceStartDay) {
        this.deviceStartDay = deviceStartDay;
    }

    public void setDeviceStartHour(int deviceStartHour) {
        this.deviceStartHour = deviceStartHour;
    }

    public void setDeviceStatus(String deviceStatus) {
        this.deviceStatus = deviceStatus;
    }

    public void setDeviceStartMinute(int deviceStartMinute) {
        this.deviceStartMinute = deviceStartMinute;
    }


    public void setDeviceModuleAccount(int deviceModuleAccount) {
        this.deviceModuleAccount = deviceModuleAccount;
    }


    public void setDeviceAddtime(Date deviceAddtime) {
        this.deviceAddtime = deviceAddtime;
    }
}
