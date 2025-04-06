package Test.example.batteryScheduling.domain;

import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Getter;

@Getter
@TableName(value = "region", autoResultMap = true)
public class Region {
    private String regionName;
    private String regionDevice;
    private String regionStatus;
    private String regionDescribe;

    public void setRegion_name(String region_name) {
        this.regionName = region_name;
    }

    public void setRegion_device(String region_device) {
        this.regionDevice = region_device;
    }

    public void setRegion_status(String region_status) {
        this.regionStatus = region_status;
    }

    public void setRegion_describe(String region_describe) {
        this.regionDescribe = region_describe;
    }
}
