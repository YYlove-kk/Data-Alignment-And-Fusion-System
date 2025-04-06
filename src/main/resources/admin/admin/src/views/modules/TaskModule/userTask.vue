<template>
  <div class="upload-component">
    <IndexHeader></IndexHeader>
    <el-button v-if=!isShowHistory @click="getHistoryData" type="info" class="showHistoryBtn">查看历史升级请求</el-button>
    <el-form v-if=!isShowHistory label-position="top" class="upload-form">

      <el-form-item label="申请人：">
        <el-input v-model="applicant" :disabled="true"></el-input>
      </el-form-item>

      <el-form-item label="主题：">
        <el-input v-model="subject" placeholder="请输入主题"></el-input>
      </el-form-item>

      <el-form-item label="申请部门：">
        <el-input v-model="department" placeholder="请输入申请部门"></el-input>
      </el-form-item>

      <el-form-item label="申请编号：">
        <el-input v-model="applicationId" :disabled="true"></el-input>
      </el-form-item>

      <el-form-item label="申请原因：">
        <el-input type="textarea" v-model="reason" placeholder="请填写授权ECR/TCR及内容"></el-input>
      </el-form-item>

      <el-row>
        <el-row>
          <el-col :span="12">
            <el-form-item label="升级包类型：">
              <el-select v-model="selectedUpgradeType" placeholder="请选择升级包类型">
                <el-option label="WIL软件" value="WIL软件"></el-option>
                <el-option label="BMS脚本" value="BMS脚本"></el-option>
              </el-select>
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="软件版本/脚本版本：">
              <el-input v-model="applicationVersion" placeholder="请输入版本号"></el-input>
            </el-form-item>
          </el-col>
        </el-row>

      </el-row>

<!--      <el-form-item label="上传文件：">-->
<!--        <el-upload-->
<!--            class="upload-area"-->
<!--            ref="upload"-->
<!--            :on-change="handleFileChange"-->
<!--            :before-upload="beforeUpload"-->
<!--            :file-list="file ? [file] : []"-->
<!--            :auto-upload="false">-->
<!--            <el-button size="small" type="primary" v-if="!file">选择文件</el-button>-->
<!--            <div slot="tip" class="el-upload__tip" v-if="!file">拖拽文件到此处或<span style="color: #409EFF;">浏览上传</span>，只接受{{ selectedUpgradeType === 'WIL软件' ? '.otap3' : '.bin' }}文件</div>-->
<!--            <div v-if="file" class="uploaded-file">-->
<!--              <span>{{ file.name }}</span>-->
<!--              <el-button size="mini" type="danger" @click="removeFile">移除</el-button>-->
<!--            </div>-->
<!--            </el-upload>-->
<!--      </el-form-item>-->

      <el-form-item label="上传文件：">
        <div v-if="!file" class="upload-area">
          <input type="file" @change="handleFileChange" :accept="selectedUpgradeType === 'WIL软件' ? '.otap3' : '.bin'">
        </div>
        <div v-if="file" class="uploaded-file">
          <span>{{ file.name }}</span>
          <el-button size="mini" type="danger" @click="removeFile">移除</el-button>
        </div>
      </el-form-item>

      <el-form-item class="form-actions">
        <el-button type="success" @click="submitAction">提交</el-button>
      </el-form-item>

    </el-form>
    <!-- 历史任务表格 -->
    <el-table v-if=isShowHistory :data="historyTasks" border>
      <el-table-column prop="applicationNumber" label="申请编号"></el-table-column>
      <el-table-column prop="applicationSubject" label="主题"></el-table-column>
      <el-table-column prop="applicationDepartment" label="部门"></el-table-column>
      <el-table-column prop="applicationReason" label="申请原因"></el-table-column>
      <el-table-column prop="upgrade_package_name" label="升级包名称">
        <template slot-scope="scope">
          <a href="javascript:;" @click="downloadFile(scope.row.applicationName, scope.row.upgrade_package_name)">
            {{ scope.row.upgrade_package_name }}
          </a>
        </template>
      </el-table-column>
      <el-table-column prop="applicationStatus" label="审核状态">
        <template v-slot:default="{ row }">
          <span :class="getStatusClass(row.applicationStatus)">
            {{ getStatusText(row.applicationStatus) }}
          </span>
        </template>
      </el-table-column>

      <el-table-column prop="applicationTimes" label="申请时间" width="180"></el-table-column>
    </el-table>

    <div style="text-align: center">
      <el-button v-if=isShowHistory @click="backUpgradeTask" type="info"  class="backHistoryBtn">返回</el-button>
    </div>

  </div>
</template>

<script>
import IndexHeader from "@/components/index/IndexHeader.vue";

export default {
  components: {
    IndexHeader,
  },
  data() {
    return {
      selectedPartNumber: '',
      selectedUpgradeType: 'WIL软件', // 初始化选择的升级包类型
      partNumbers: [
        { value: '24120503', label: '24120503' },
        { value: '24120504', label: '24120504' },
        { value: '24120149', label: '24120149' },
        { value: '24120150', label: '24120150' },
        { value: '24120443', label: '24120443' },
        { value: '24120444', label: '24120444' },
      ],
      file: null, // 存储单个文件
      subject: '',
      department: '',
      applicant: this.$storage.get('adminName'), // 获取申请人名称
      applicationId: '',
      applicationVersion:'',
      reason: '',
      uploadProgress: 0, // 用于追踪上传进度
      historyTasks: [], // 用于存储历史任务数据
      isShowHistory:false, // 用于展示历史信息
    };
  },
  mounted() {
    this.fetchApplicationId();
  },
  methods: {
    async fetchApplicationId() {
      try {
        const response = await this.$http.get("UpgradeTask/getRandomUpgradeTaskNumber");
        const data = response.data;

        if (data && data.status === 0) {
          this.applicationId = data.data;
        } else {
          this.applicationId = '';
        }
      } catch (error) {
        console.error("Failed to fetch device data:", error);
      } finally {
        this.dataListLoading = false;
      }
    },
    async getHistoryData() {
      this.isShowHistory=true;
      const username = this.$storage.get('adminName');
      console.log("adminName:" + username);
      try {
        const response = await this.$http.get("UpgradeTask/getUpgradeTaskHistoryByUserName", {
          params: { username }
        });
        const data = response.data;

        if (data && data.status === 0) {
          this.historyTasks = data.data || [];
        } else {
          this.$message.error("无法获取历史数据");
        }
      } catch (error) {
        console.error("获取历史数据时出现错误:", error);
        this.$message.error("获取历史数据失败");
      }
    },
    backUpgradeTask() {
      this.isShowHistory = false;
    },

    removeFile() {
      this.file = null;
      this.$refs.upload.clearFiles(); // 清除 el-upload 组件的文件列表
    },


    handleFileChange(event) {
      const files = event.target.files;
      if (files.length > 0) {
        const file = files[0];
        const fileExtension = file.name.split('.').pop().toLowerCase();
        const acceptableExtensions = this.selectedUpgradeType === 'WIL软件' ? ['otap3'] : ['bin'];
        if (acceptableExtensions.includes(fileExtension)) {
          this.file = file;
        } else {
          this.$message.error(`只接受${this.selectedUpgradeType === 'WIL软件' ? '.otap3' : '.bin'}文件`);
        }
      }
    },


    addFile(file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        // 创建一个与 Element UI file 对象格式相似的对象
        const fileObj = {
          name: file.name,
          url: e.target.result,
          status: 'ready'
        };
        this.fileList.push(fileObj);
      };
      reader.readAsDataURL(file);
    },

    beforeRemove(file) {
      return this.$confirm(`确认移除 ${file.name}？`)
          .then(() => {
            this.file = null;
            this.$refs.upload.clearFiles(); // 清除 el-upload 组件的文件列表
            return true;
          })
          .catch(() => false); // 如果用户取消，阻止文件删除
    },

    beforeUpload(file) {
      const fileExtension = file.name.split('.').pop().toLowerCase();
      const acceptableExtensions = this.selectedUpgradeType === 'WIL软件' ? ['otap3'] : ['bin'];
      if (!acceptableExtensions.includes(fileExtension)) {
        this.$message.error(`只接受 ${this.selectedUpgradeType === 'WIL软件' ? '.otap3' : '.bin'} 文件类型`);
        return false;
      }
      return true;
    },


    confirmAction() {
      // 确认操作，可根据需要实现
      console.log("确认操作");
    },
    formatDate(date) {
      const pad = (s) => (s < 10 ? '0' + s : s);
      return (
          date.getFullYear() +
          '-' +
          pad(date.getMonth() + 1) +
          '-' +
          pad(date.getDate()) +
          ' ' +
          pad(date.getHours()) +
          ':' +
          pad(date.getMinutes()) +
          ':' +
          pad(date.getSeconds())
      );
    },

    submitAction() {
      console.log("用户提交操作：  " + this.$storage.get('adminName'));
      if (!this.department) {
        this.$message.error('请填写申请部门');
        return;
      }
      if (!this.reason) {
        this.$message.error('请填写申请原因');
        return;
      }
      if (!this.file) {
        this.$message.error('请先选择一个升级文件包');
        return;
      }
      if (!this.applicationVersion) {
        this.$message.error('请填写软件版本/脚本版本');
        return;
      }

      if (this.file) {
        const formData = new FormData();
        formData.append('file', this.file);
        formData.append('userName', this.$storage.get('adminName'));

        // 使用 this.$http,上传文件
        this.$http.post("UpgradeFile/uploadFiles", formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        }).then(response => {
          const data = response.data;
          if (data && data.status === 0) {
            console.log("上传文件成功");
            this.$message({
              message: '上传成功',
              type: 'success'
            });
            // 清空文件列表
            this.file = null;
          } else {
            this.$message({
              message: data.message || "上传失败",
              type: 'error'
            });
          }
        }).catch(error => {
          // 处理错误
          this.$message.error('上传文件过程中发生错误');
          console.error("上传过程中发生错误:", error);
          return;
        });

        const upgradeTaskAdmitDTO = {
          applicationNumber: this.applicationId,
          applicationName: this.applicant,
          applicationSubject: this.subject,
          applicationDepartment: this.department,
          applicationReason: this.reason,
          upgrade_package_name: this.file.name,
          application_part_number: this.selectedPartNumber,
          application_type: this.selectedUpgradeType, // 请根据需要设置正确的值
          applicationVersion:this.applicationVersion,
          applicationTimes: this.formatDate(new Date()) // 由前端设置当前时间
        };
        // 使用 this.$http 发送请求
        this.$http.post("UpgradeTask/addUpgradeTask", upgradeTaskAdmitDTO)
            .then(response => {
              const data = response.data;
              if (data && data.status === 0) {
                this.$message({
                  message: '升级任务上传成功',
                  type: 'success'
                });
                // 清空表单
                this.resetForm();
              } else {
                this.$message({
                  message: data.message || "记录更新失败",
                  type: 'error'
                });
              }
            })
            .catch(error => {
              console.error("升级任务上传过程中发生错误:", error);
              this.$message({
                message: "升级任务上传过程中发生错误",
                type: 'error'
              });
            });
      }else{
        this.$message.error('请先选择一个升级文件包');
      }
    },

    resetForm() {
      this.subject = '';
      this.department = '';
      this.reason = '';
      this.file = null;
      this.fetchApplicationId(); // 重新获取申请编号
    },

    getStatusText(status) {
      const statusMap = {
        '0': '待审核',
        '1': '已通过',
        '2': '已拒绝',
      };
      return statusMap[status] || '未知';
    },
    getStatusClass(status) {
      const statusClassMap = {
        '1': 'status-accepted',
        '2': 'status-rejected',
      };
      return statusClassMap[status] || '';
    },
    //文件下载
    downloadFile(applicationName, upgradePackageName) {
      this.$confirm('是否下载文件?', '提示', {
        confirmButtonText: '下载',
        cancelButtonText: '取消',
        type: 'warning'
      }).then(() => {
        const params = new URLSearchParams();
        params.append('applicationName', applicationName);
        params.append('upgradePackageName', upgradePackageName);

        // 使用 Axios 发送 GET 请求，注意设置 responseType 为 'blob'
        this.$http.get('/UpgradeFile/getUpgradeByFileNameAndUserName', {
          params: {
            applicationName: applicationName,
            upgradePackageName: upgradePackageName
          },
          responseType: 'json'  // Expecting a JSON response
        }).then(response => {
          if(response.data.status === 0) {
            const result = response.data.data;
            const base64Data = result.fileContent;
            // Assuming fileContent is the Base64 encoded string
            const blob = this.base64ToBlob(base64Data, "application/octet-stream"); // You might want to adjust the MIME type
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.setAttribute('download', upgradePackageName);
            document.body.appendChild(link);
            link.click();
            link.remove();
          } else {
            this.$message.error('下载失败: ' + response.data.message);
          }
        }).catch(error => {
          console.error("下载错误", error);
          this.$message.error('文件下载失败');
        });
      });
    },
    base64ToBlob(base64, mime) {
      const byteChars = atob(base64);
      const byteNumbers = new Array(byteChars.length);
      for (let i = 0; i < byteChars.length; i++) {
        byteNumbers[i] = byteChars.charCodeAt(i);
      }
      const byteArray = new Uint8Array(byteNumbers);
      return new Blob([byteArray], {type: mime});
    },

  }
};
</script>

<style scoped>

.showHistoryBtn{
  height: 40px;
  color: #333;
  font-size: 14px;
  border-width: 1px;
  border-style: solid;
  border-color: #DCDFE6;
  border-radius: 4px;
  background-color: rgb(113, 249, 229);
}

.backHistoryBtn{
  height: 40px;
  color: #333;
  font-size: 14px;
  border-width: 1px;
  border-style: solid;
  border-color: #DCDFE6;
  border-radius: 4px;
  background-color: rgb(113, 249, 229);
}

.status-accepted {
  color: green;
}

.status-rejected {
  color: red;
}

.upload-component {
  padding: 20px;
}

.upload-form {
  max-width: 600px;
  margin: 0 auto;
}

.upload-area {
  border: 2px dashed #d9d9d9;
  border-radius: 5px;
  background-color: #fafafa;
  padding: 15px;
  text-align: center;
}

.uploaded-file {
  margin-top: 10px;
  align-items: center;
  display: flex;
  justify-content: space-between;
}


.form-actions {
  display: flex;
  justify-content: space-between;
}


/* 响应式布局适配 */
@media (max-width: 768px) {
  .upload-form {
    padding: 10px;
  }

  .upload-form .el-form-item {
    margin-bottom: 20px; /* 增加表单项的底部间距 */
  }

  .upload-area {
    padding: 10px; /* 减小上传区域的内边距 */
  }

  .uploaded-file {
    flex-direction: column; /* 文件名和操作按钮垂直排列 */
    gap: 5px; /* 增加间距 */
  }

  .form-actions {
    flex-direction:row; /* 确保按钮垂直排列 */
    width: 100%;
    align-content: center;
  }


}

</style>
