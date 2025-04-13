package DataAlignmentAndFusionApplication.model.enums;

import lombok.Getter;

//final不用提供set方法
@Getter
public enum ResponseCode {

    SUCCESS(0,"SUCCESS"),
    ERROR(1,"ERROR");

    private final int code;
    private final String desc;

    ResponseCode(int code, String desc) {
        this.code = code;
        this.desc = desc;
    }
}
