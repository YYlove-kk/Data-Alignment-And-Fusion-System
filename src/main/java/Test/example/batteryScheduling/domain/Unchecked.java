package Test.example.batteryScheduling.domain;

import com.baomidou.mybatisplus.annotation.TableName;
import com.fasterxml.jackson.annotation.JsonFormat;
import lombok.Data;
import lombok.Getter;
import org.springframework.format.annotation.DateTimeFormat;

import java.util.Date;

@Getter
@Data
@TableName(value = "unchecked", autoResultMap = true)
public class Unchecked {


    /*
        托号
     */
    private String setNumber;//
    /*
        状态
     */
    private String setStatus;//
    /*
            分区
         */
    private String setReason;//
    /*
            模组电池型号
         */
    @JsonFormat(locale="zh", timezone="GMT+8", pattern="yyyy-MM-dd HH:mm:ss")
    @DateTimeFormat
    private Date add_time;

    public void setSetNumber(String setNumber) {
        this.setNumber = setNumber;
    }

    public void setSetStatus(String setStatus) {
        this.setStatus = setStatus;
    }

    public void setSetReason(String setReason) {
        this.setReason = setReason;
    }

    public void setAdd_time(Date add_time) {
        this.add_time = add_time;
    }
}
