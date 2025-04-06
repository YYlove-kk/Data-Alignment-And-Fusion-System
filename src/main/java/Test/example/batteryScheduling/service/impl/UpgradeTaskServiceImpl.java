package Test.example.batteryScheduling.service.impl;

import Test.example.batteryScheduling.common.CommonResponse;
import Test.example.batteryScheduling.domain.UpgradeTask;
import Test.example.batteryScheduling.domain.User;
import Test.example.batteryScheduling.persistance.UpgradeTaskMapper;
import Test.example.batteryScheduling.service.UpgradeTaskService;
import Test.example.batteryScheduling.service.UserService;
import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import Test.example.batteryScheduling.DTO.UpgradeTaskAdmitDTO;
import Test.example.batteryScheduling.DTO.UpgradeTaskDTO;
import org.springframework.core.io.Resource;
import org.springframework.beans.BeanUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.UrlResource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.mail.SimpleMailMessage;
import org.springframework.mail.javamail.JavaMailSender;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.io.File;
import java.io.IOException;
import java.net.MalformedURLException;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;
import java.util.Random;
import java.util.TimeZone;
import java.util.stream.Collectors;

@Validated
@CrossOrigin
@RestController
@RequestMapping("/UpgradeTask")
@Service("upgradeTaskService")
public class UpgradeTaskServiceImpl extends ServiceImpl<UpgradeTaskMapper, UpgradeTask> implements UpgradeTaskService {

    @Autowired
    private JavaMailSender mailSender;

    @Autowired
    private UpgradeTaskMapper upgradeTaskMapper;

    @Autowired
    private UserService userService;
    private String uploadDir = "src/main/resources/UpgradeFileDir/";

    @Override
    public CommonResponse<List<UpgradeTaskDTO>> getAllUpgradeTask() {
        // 构建查询条件，只选择applicationStatus为"0"的数据
        QueryWrapper<UpgradeTask> wrapper = new QueryWrapper<>();
        wrapper.eq("application_status", "0");
        List<UpgradeTask> upgradeTaskList = list(wrapper);

        // 转换为UpgradeTaskDTO列表
        List<UpgradeTaskDTO> upgradeTaskDTOList = upgradeTaskList.stream().map(upgradeTask -> {
            UpgradeTaskDTO upgradeTaskDTO = new UpgradeTaskDTO();
            BeanUtils.copyProperties(upgradeTask, upgradeTaskDTO);
//            upgradeTaskDTO.setApplicationType(upgradeTask.getApplicationType());
            return upgradeTaskDTO;
        }).collect(Collectors.toList());

        if(!upgradeTaskDTOList.isEmpty()) {
            return CommonResponse.createForSuccess("获取未审核的上传任务成功", upgradeTaskDTOList);
        } else {
            return CommonResponse.createForError("未找到未审核的上传任务");
        }
    }

    @Override
    @Transactional
    public CommonResponse<Boolean> addUpgradeTask(UpgradeTaskAdmitDTO upgradeTaskAdmitDTO) {
            UpgradeTask upgradeTask = new UpgradeTask();
        BeanUtils.copyProperties(upgradeTaskAdmitDTO, upgradeTask);
        boolean success = false;
        // 设置状态为0——未审核
        upgradeTask.setApplicationStatus("0");
        // 设置申请时间为当前时间
        upgradeTask.setApplicationTimes(new Date());
        try {
            success = save(upgradeTask); // Use MyBatis Plus 'save' method
        } catch (Exception e) {
            // Logging the exception
            e.printStackTrace();
            return CommonResponse.createForError("Failed to add upgrade task due to: " + e.getMessage());
        }
        return success ? CommonResponse.createForSuccess(true) : CommonResponse.createForError("Failed to save upgrade task.");
    }

    @Override
    @Transactional
    public CommonResponse<String> getRandomUpgradeTaskNumber() {
        String applicationNumber;
        boolean exists;

        do {
            applicationNumber = generateRandomApplicationNumber(10);
            exists = checkApplicationNumberExists(applicationNumber);
        } while (exists);

        System.out.println("APN:" + applicationNumber);
        return CommonResponse.createForSuccess(applicationNumber);
    }

    @Override
    @Transactional
    public CommonResponse<String> rejectUpgradeTask(UpgradeTaskDTO upgradeTaskDTO,String rejectReason) throws ParseException {
        UpgradeTask existingTask = upgradeTaskMapper.selectById(upgradeTaskDTO.getApplicationNumber());

        if (existingTask  == null) {
            return CommonResponse.createForError("No task found with applicationNumber: " + upgradeTaskDTO.getApplicationNumber());
        }
        // 使用SimpleDateFormat来格式化和解析日期
        SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
        dateFormat.setTimeZone(TimeZone.getTimeZone("GMT+8")); // 设置时区为GMT+8
        // 获取当前时间的Date对象
        Date now = new Date();
        // 将当前时间格式化为字符串，然后再解析回Date对象（这一步实际上是多余的，直接使用now即可）
        String formattedDate = dateFormat.format(now);
        Date parsedDate;
        try {
            parsedDate = dateFormat.parse(formattedDate);
        } catch (ParseException e) {
            e.printStackTrace();
            return CommonResponse.createForError("解析日期失败: " + e.getMessage());
        }
        existingTask.setApplicationStatus("2"); // 设置状态为1表示拒绝并下发
        existingTask.setApplicationTimes(parsedDate); // 使用解析回的Date对象
        // 更新记录
        int updated = upgradeTaskMapper.updateById(existingTask);

        if (updated > 0) {
            //发送邮件
            //根据用户名获取用户
            User user = userService.getUserByUserName(upgradeTaskDTO.getApplicationName());
            String email = user.getEmail();
            //将rejectReason内容发送给email
            sendEmail(email, "升级任务被拒绝", rejectReason);

            return CommonResponse.createForSuccess("Task rejected successfully");
        } else {
            return CommonResponse.createForError("Failed to reject task");
        }
    }

    @Override
    public CommonResponse<String> agreeUpgradeTask(UpgradeTaskDTO upgradeTaskDTO) {
        UpgradeTask existingTask = upgradeTaskMapper.selectById(upgradeTaskDTO.getApplicationNumber());
        if (existingTask == null) {
            return CommonResponse.createForError("No task found with applicationNumber: " + upgradeTaskDTO.getApplicationNumber());
        }

        existingTask.setApplicationStatus("1"); // 设置状态为1表示同意并下发
        existingTask.setApplicationTimes(new Date()); // 直接使用当前时间

        // Define file path based on applicationType
        String baseDir = "src/main/resources/UpgradeFile/";
        String newFileName = upgradeTaskDTO.getApplicationType().equals("WIL软件") ? "CMU_OPFW.otap3" : "BMS_CN.bin";
        Path sourceFilePath = Paths.get(uploadDir, upgradeTaskDTO.getApplicationName(), upgradeTaskDTO.getUpgrade_package_name());
        Path targetFilePath = Paths.get(baseDir, newFileName);

        try {
            // Ensure directory exists
            Files.createDirectories(targetFilePath.getParent());

            // Check if source file exists and copy it to new location with new filename
            if (Files.exists(sourceFilePath)) {
                Files.copy(sourceFilePath, targetFilePath, StandardCopyOption.REPLACE_EXISTING);

                // Update version.txt file
                Path versionFilePath = Paths.get(baseDir, "version.txt");
                updateVersionFile(versionFilePath, upgradeTaskDTO);

                int updated = upgradeTaskMapper.updateById(existingTask);
                if (updated > 0) {
                    User user = userService.getUserByUserName(upgradeTaskDTO.getApplicationName());
                    String email = user.getEmail();
                    sendEmail(email, "请求审批通知", existingTask.getApplicationName() + "的升级包为" + existingTask.getUpgrade_package_name() + "的升级任务审批已通过");

                    return CommonResponse.createForSuccess("升级任务请求通过表更新成功，并且文件已成功复制及重命名");
                } else {
                    return CommonResponse.createForError("升级任务请求通过表更新失败");
                }
            } else {
                return CommonResponse.createForError("原始文件不存在");
            }
        } catch (IOException e) {
            e.printStackTrace();
            return CommonResponse.createForError("文件操作失败: " + e.getMessage());
        } catch (Exception e) {
            e.printStackTrace();
            return CommonResponse.createForError("硬件升级任务下发失败");
        }
    }

    private void updateVersionFile(Path versionFilePath, UpgradeTaskDTO upgradeTaskDTO) throws IOException {
        List<String> lines = Files.readAllLines(versionFilePath, StandardCharsets.UTF_8);
        if (upgradeTaskDTO.getApplicationType().equals("WIL软件")) {
            lines.set(0, upgradeTaskDTO.getApplicationVersion());
        } else {
            lines.set(1, upgradeTaskDTO.getApplicationVersion());
        }
        Files.write(versionFilePath, lines, StandardCharsets.UTF_8, StandardOpenOption.TRUNCATE_EXISTING);
    }



    //获取更新任务历史信息
    @Override
    public CommonResponse<List<UpgradeTaskDTO>> getUpgradeTaskHistory() {
        // 构建查询条件，选择applicationStatus为非"0"的数据
        QueryWrapper<UpgradeTask> wrapper = new QueryWrapper<>();
        wrapper.ne("application_status", "0");
        List<UpgradeTask> upgradeTaskList = list(wrapper);

        // 转换为UpgradeTaskDTO列表
        List<UpgradeTaskDTO> upgradeTaskDTOList = upgradeTaskList.stream().map(upgradeTask -> {
            UpgradeTaskDTO upgradeTaskDTO = new UpgradeTaskDTO();
            BeanUtils.copyProperties(upgradeTask, upgradeTaskDTO);
            return upgradeTaskDTO;
        }).collect(Collectors.toList());

        if(!upgradeTaskDTOList.isEmpty()) {
            return CommonResponse.createForSuccess("获取未审核的上传任务成功", upgradeTaskDTOList);
        } else {
            return CommonResponse.createForError("未找到未审核的上传任务");
        }
    }

    @Override
    public CommonResponse<List<UpgradeTaskDTO>> getUpgradeTaskHistoryByUserName(String username) {
        QueryWrapper<UpgradeTask> wrapper = new QueryWrapper<>();
        wrapper.eq("application_name", username);
        List<UpgradeTask> upgradeTaskList = list(wrapper);
        System.out.println("HIStory userName" + username);

        // 转换为UpgradeTaskDTO列表
        List<UpgradeTaskDTO> upgradeTaskDTOList = upgradeTaskList.stream().map(upgradeTask -> {
            UpgradeTaskDTO upgradeTaskDTO = new UpgradeTaskDTO();
            BeanUtils.copyProperties(upgradeTask, upgradeTaskDTO);
            return upgradeTaskDTO;
        }).collect(Collectors.toList());

        if(!upgradeTaskDTOList.isEmpty()) {
            return CommonResponse.createForSuccess("获取用户已上传任务成功", upgradeTaskDTOList);
        } else {
            return CommonResponse.createForError("未找到用户上传任务");
        }
    }

    @Override
    public ResponseEntity<Resource> getUpgradeFileBMS() {
        try {
            Path filePath = Paths.get("src/main/resources/UpgradeFile/BMS_CN.bin");
            Resource resource = new UrlResource(filePath.toUri());

            if (resource.exists() && resource.isReadable()) {
                return ResponseEntity.ok()
                        .header(HttpHeaders.CONTENT_DISPOSITION, "attachment; filename=\"" + resource.getFilename() + "\"")
                        .contentType(MediaType.APPLICATION_OCTET_STREAM)
                        .body(resource);
            } else {
                return ResponseEntity.badRequest()
                        .body(null);
            }
        } catch (MalformedURLException e) {
            return ResponseEntity.internalServerError().build();
        }
    }

    @Override
    public ResponseEntity<Resource> getUpgradeFileWIL() {
        try {
            Path filePath = Paths.get("src/main/resources/UpgradeFile/CMU_OPFW.otap3");
            Resource resource = new UrlResource(filePath.toUri());

            if (resource.exists() && resource.isReadable()) {
                return ResponseEntity.ok()
                        .header(HttpHeaders.CONTENT_DISPOSITION, "attachment; filename=\"" + resource.getFilename() + "\"")
                        .contentType(MediaType.APPLICATION_OCTET_STREAM)
                        .body(resource);
            } else {
                return ResponseEntity.badRequest()
                        .body(null);
            }
        } catch (MalformedURLException e) {
            return ResponseEntity.internalServerError().build();
        }
    }

    @Override
    public ResponseEntity<Resource> getUpgradeFileVersion() {
        try {
            Path filePath = Paths.get("src/main/resources/UpgradeFile/version.txt");
            Resource resource = new UrlResource(filePath.toUri());

            if (resource.exists() && resource.isReadable()) {
                return ResponseEntity.ok()
                        .header(HttpHeaders.CONTENT_DISPOSITION, "attachment; filename=\"" + resource.getFilename() + "\"")
                        .contentType(MediaType.APPLICATION_OCTET_STREAM)
                        .body(resource);
            } else {
                return ResponseEntity.badRequest()
                        .body(null);
            }
        } catch (MalformedURLException e) {
            return ResponseEntity.internalServerError().build();
        }
    }

    //生成length位随机编号，随机的数字、小写字母和大写字母的组合
    private String generateRandomApplicationNumber(int length) {
        String characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
        StringBuilder result = new StringBuilder();
        Random random = new Random();

        for (int i = 0; i < length; i++) {
            int index = random.nextInt(characters.length());
            result.append(characters.charAt(index));
        }

        return result.toString();
    }

    //检查编号是否已经存在于数据库中
    private boolean checkApplicationNumberExists(String applicationNumber) {
        return this.lambdaQuery().eq(UpgradeTask::getApplicationNumber, applicationNumber).count() > 0;
    }

    //发送邮件
    private void sendEmail(String to, String subject, String text) {
        SimpleMailMessage message = new SimpleMailMessage();
        message.setFrom("2570089991@qq.com"); // 发件人邮箱
        message.setTo(to);
        message.setSubject(subject);
        message.setText(text);
        mailSender.send(message);
    }

}
