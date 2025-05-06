package DataAlignmentAndFusionApplication.common.enums;

import lombok.Getter;

@Getter
public enum UploadTpye {
    TEXT(0,"TEXT"),
    IMAGE(1,"IMAGE");

    private final int code;
    private final String desc;

    UploadTpye(int code, String desc) {
        this.code = code;
        this.desc = desc;
    }
}
