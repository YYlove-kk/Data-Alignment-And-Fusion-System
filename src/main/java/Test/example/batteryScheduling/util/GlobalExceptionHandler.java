package Test.example.batteryScheduling.util;


import Test.example.batteryScheduling.common.CommonResponse;
import jakarta.validation.ConstraintViolationException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.MissingServletRequestParameterException;
import org.springframework.web.bind.annotation.ControllerAdvice;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.method.annotation.MethodArgumentTypeMismatchException;

@ControllerAdvice//切面切入每个Controller
public class GlobalExceptionHandler {

    Logger logger = LoggerFactory.getLogger(GlobalExceptionHandler.class);

    //使后端错误可以被前端认识
    //全局异常处理，之后任何的需要参数的方法在缺少参数时都返回这个错误信息
    @ExceptionHandler(MissingServletRequestParameterException.class)
    @ResponseStatus(code = HttpStatus.BAD_REQUEST)
    @ResponseBody
    public CommonResponse<Object> handleMissingServletRequestParameterException(MissingServletRequestParameterException e){
        logger.error(e.getMessage());
        return CommonResponse.createForError("缺少参数");
    }

    @ExceptionHandler(MethodArgumentNotValidException.class)
    @ResponseStatus(code = HttpStatus.BAD_REQUEST)
    @ResponseBody
    public CommonResponse<Object> handleMethodArgumentNotValidException(MethodArgumentNotValidException e){
        logger.error(e.getMessage());
        return CommonResponse.createForError("参数不合法");
    }


    @ExceptionHandler(MethodArgumentTypeMismatchException.class)
    @ResponseStatus(code = HttpStatus.BAD_REQUEST)
    @ResponseBody
    public CommonResponse<Object> handleMethodArgumentTypeMismatchException(MethodArgumentTypeMismatchException e){
        logger.error(e.getMessage());
        return CommonResponse.createForError("参数类型错误");
    }


    @ExceptionHandler(ConstraintViolationException.class )
    @ResponseStatus(code = HttpStatus.INTERNAL_SERVER_ERROR)
    @ResponseBody
    public CommonResponse<Object> handleConstraintViolationException(ConstraintViolationException e){
        logger.error(e.getMessage());
        return CommonResponse.createForError(e.getMessage());
    }

    //异常从上往下一次匹配，当之前的所有异常都没有匹配时匹配这个异常
    //存在继承关系的异常时父类要写在后面，否则匹配到子类之前将直接与父类匹配
    @ExceptionHandler(Exception.class)
    @ResponseStatus(code = HttpStatus.INTERNAL_SERVER_ERROR)
    @ResponseBody
    public CommonResponse<Object> handleException(Exception e){
        logger.error(e.getMessage());
        e.printStackTrace();
        return CommonResponse.createForError("服务器异常");
    }

}















