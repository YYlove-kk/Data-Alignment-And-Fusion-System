package Test.example.batteryScheduling.service.impl;

import Test.example.batteryScheduling.common.CommonResponse;
import Test.example.batteryScheduling.persistance.UpgradeFileMapper;
import Test.example.batteryScheduling.service.UpgradeFileService;
import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import Test.example.batteryScheduling.domain.UpgradeFile;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.*;

@Validated
@CrossOrigin
@RestController
@RequestMapping("/UpgradeFile")
@Service("upgradeFileService")
public class UpgradeFileServiceImpl extends ServiceImpl<UpgradeFileMapper, UpgradeFile> implements UpgradeFileService {
    @Autowired
    private UpgradeFileMapper upgradeFileMapper;

    private String uploadDir = "src/main/resources/UpgradeFileDir/";

    @Override
    @Transactional
    public CommonResponse<Boolean> handleFilesUpload(MultipartFile file, String userName) {
            try {
                String originalFilename = file.getOriginalFilename();
                // Ensure the directory exists
                String userUploadDir = uploadDir + userName;
                File directory = new File(userUploadDir);
                if (!directory.exists()) directory.mkdirs();

                // Resolve the filename in case of duplicates
                String fileName = resolveFileName(originalFilename, userName, userUploadDir);
                File targetFile = new File(userUploadDir, fileName);

                // Save the file
                Files.copy(file.getInputStream(), targetFile.toPath(), StandardCopyOption.REPLACE_EXISTING);

                // Record in database
                UpgradeFile existingFile = upgradeFileMapper.selectOne(new QueryWrapper<UpgradeFile>()
                        .eq("user_name", userName)
                        .eq("file_name", fileName));

                if (existingFile == null) {
                    // 保存新纪录至数据库
                    UpgradeFile upgradeFile = new UpgradeFile();
                    upgradeFile.setUserName(userName);
                    upgradeFile.setFileName(fileName);
                    upgradeFile.setPath(userUploadDir);
                    upgradeFileMapper.insert(upgradeFile);
                } else {
                    // 文件已存在，更新路径
                    if (!existingFile.getPath().equals(userUploadDir)) {
                        existingFile.setPath(userUploadDir);
                        upgradeFileMapper.updateById(existingFile);
                    }
                }

            } catch (Exception e) {
                e.printStackTrace();
                return CommonResponse.createForError("Failed to upload file: " + e.getMessage());
            }
        return CommonResponse.createForSuccess(true);
    }



    private String resolveFileName(String originalFilename, String userName, String basePath) {
        int counter = 0;
        String fileName = originalFilename;
        while (new File(basePath + fileName).exists()) {
            counter++;
            // 分离文件名和扩展名
            int dotIndex = originalFilename.lastIndexOf(".");
            String nameWithoutExtension = (dotIndex == -1) ? originalFilename : originalFilename.substring(0, dotIndex);
            String extension = (dotIndex == -1) ? "" : originalFilename.substring(dotIndex + 1);
            fileName = String.format("%s(%d).%s", nameWithoutExtension, counter, extension);
        }
        return fileName;
    }


    @Override
    public CommonResponse<Map<String, String>> getUpgradeByFileNameAndUserName(String applicationName, String upgradePackageName) {
        File file = new File(uploadDir + applicationName + "/" + upgradePackageName);
        if (!file.exists()) {
            return CommonResponse.createForError("File not found");
        }

        try {
            byte[] data = Files.readAllBytes(file.toPath());
            String base64Encoded = Base64.getEncoder().encodeToString(data);

            Map<String, String> result = new HashMap<>();
            result.put("fileName", upgradePackageName);
            result.put("fileContent", base64Encoded);

            return CommonResponse.createForSuccess(result);
        } catch (IOException e) {
            return CommonResponse.createForError(e.getMessage());
        }
    }

}
