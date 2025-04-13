package DataAlignmentAndFusionApplication.util;

import lombok.Data;

@Data
public class Result<T> {
    private int code;    // 状态码（如200成功，500失败）
    private String msg;  // 提示信息
    private T data;      // 返回的数据（泛型）

    // 成功静态方法
    public static <T> Result<T> success(T data) {
        Result<T> result = new Result<>();
        result.setCode(200);
        result.setMsg("成功");
        result.setData(data);
        return result;
    }

    // 失败静态方法
    public static <T> Result<T> error(int code, String msg) {
        Result<T> result = new Result<>();
        result.setCode(code);
        result.setMsg(msg);
        return result;
    }
}
