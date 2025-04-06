package Test.example.batteryScheduling.domain;

import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import com.fasterxml.jackson.annotation.JsonFormat;
import lombok.Data;
import lombok.Getter;
import org.springframework.format.annotation.DateTimeFormat;

import java.util.Date;

@Getter
@Data
@TableName(value = "stock", autoResultMap = true)
public class Stock {


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
    private String divideRegion;//
    /*
            模组电池型号
         */
    private String moduleCategory;//

    public void setSetNumber(String setNumber) {
        this.setNumber = setNumber;
    }

    public void setSetStatus(String setStatus) {
        this.setStatus = setStatus;
    }

    public void setDivideRegion(String divideRegion) {
        this.divideRegion = divideRegion;
    }

    public void setModuleCategory(String moduleCategory) {
        this.moduleCategory = moduleCategory;
    }
}
