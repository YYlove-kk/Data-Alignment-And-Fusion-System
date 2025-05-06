package DataAlignmentAndFusionApplication.common.enums;

import lombok.Getter;

@Getter
public enum BuildMode {

    MULTI(0,"MULTI"),
    SINGLE(1,"SINGLE");

    private final int code;
    private final String desc;

    BuildMode(int code, String desc) {
        this.code = code;
        this.desc = desc;
    }
}
