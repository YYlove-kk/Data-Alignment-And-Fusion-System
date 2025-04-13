package DataAlignmentAndFusionApplication.common;

import DataAlignmentAndFusionApplication.model.enums.ResponseCode;
import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.Getter;

//保证客户端拿到数据都是同样的格式
@Getter
@JsonInclude(JsonInclude.Include.NON_NULL)//设置空值不序列化
public class CommonResponse<T> {
    private final int status;//状态:成功or失败(与前端约定好)
    private final String message;//消息
    private T data;//返回数据

    private   CommonResponse(int status,String message){
        this.status = status;
        this.message = message;
    }

    private CommonResponse(int status,String message,T data){
        this.status = status;
        this.message = message;
        this.data = data;
    }

    @JsonIgnore//默认为所有public方法序列化，打上此注解后则忽略
    public boolean isSuccess(){
        return this.status == ResponseCode.SUCCESS.getCode();
    }

    public static <T>CommonResponse<T> createForSuccess(){
        return new CommonResponse<>(ResponseCode.SUCCESS.getCode(),ResponseCode.SUCCESS.getDesc());
    }

    public static <T>CommonResponse<T> createForSuccessMessage(String message){
        return new CommonResponse<>(ResponseCode.SUCCESS.getCode(),message);
    }

    public static <T>CommonResponse<T> createForSuccess(T data){
        return new CommonResponse<>(ResponseCode.SUCCESS.getCode(),ResponseCode.SUCCESS.getDesc(),data);
    }

    //返回状态（0/1），成功提示消息以及数据----常用
    public static <T>CommonResponse<T> createForSuccess(String message,T data){
        return new CommonResponse<>(ResponseCode.SUCCESS.getCode(),message,data);
    }

    public static <T>CommonResponse<T> createForError(){
        return new CommonResponse<>(ResponseCode.ERROR.getCode(),ResponseCode.ERROR.getDesc());
    }

    //返回错误信息
    public static <T>CommonResponse<T> createForError(String message){
        return new CommonResponse<>(ResponseCode.ERROR.getCode(),message);
    }

    public static <T>CommonResponse<T> createForError(int code,String message){
        return new CommonResponse<>(code,message);
    }


}
