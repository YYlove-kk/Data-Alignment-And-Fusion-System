package Test.example.batteryScheduling.DTO;

import com.baomidou.mybatisplus.annotation.TableId;
import com.fasterxml.jackson.annotation.JsonFormat;
import lombok.Getter;
import org.springframework.format.annotation.DateTimeFormat;

import java.util.Date;

@Getter
public class BatteryCellDTO {
    /**
     * 主键id
     */
    @TableId
    private Long id;

    /*
        电芯数量
     */

    private String batteryCellNumber;

    /*
        电芯电压
     */
    private Double batteryVolt;

    /*
        电芯所在模组MAC地址
     */
    private String macAddress;

    public void setId(Long id) {
        this.id = id;
    }

    public void setBatteryCellNumber(String batteryCellNumber) {
        this.batteryCellNumber = batteryCellNumber;
    }

    public void setBatteryVolt(Double batteryVolt) {
        this.batteryVolt = batteryVolt;
    }

    public void setMacAddress(String macAddress) {
        this.macAddress = macAddress;
    }
}
