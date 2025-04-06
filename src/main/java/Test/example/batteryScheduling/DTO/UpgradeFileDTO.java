package Test.example.batteryScheduling.DTO;


public class UpgradeFileDTO {
    private Long id;
    /*
        上传人名称
     */
    private String userName;//

    /*
        文件名称
     */
    private String fileName;

    /*
        文件相对路径
     */
    private String path;

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getUserName() {
        return userName;
    }

    public void setUserName(String userName) {
        this.userName = userName;
    }

    public String getFileName() {
        return fileName;
    }

    public void setFileName(String fileName) {
        this.fileName = fileName;
    }

    public String getPath() {
        return path;
    }

    public void setPath(String path) {
        this.path = path;
    }
}
