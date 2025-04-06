<template>
  <div>
    <IndexHeader></IndexHeader>
  <div class="task-request-component">
    <el-form :inline="true" :model="searchForm" class="search-form">
      <el-form-item label="申请人">
        <el-input v-model="searchForm.applicationName" placeholder="请输入申请人"></el-input>
      </el-form-item>

      <el-form-item label="申请部门">
        <el-input v-model="searchForm.applicationDepartment" placeholder="请输入申请部门"></el-input>
      </el-form-item>

      <el-form-item label="申请编号">
        <el-input v-model="searchForm.applicationNumber" placeholder="请输入申请编号"></el-input>
      </el-form-item>

      <el-form-item>
        <el-button type="primary" @click="searchTasks">搜索</el-button>
      </el-form-item>

      <el-form-item class="history-toggle">
        <el-button type="info" @click="toggleHistory">{{ showHistory ? '显示待审核任务' : '显示历史信息状态' }}</el-button>
      </el-form-item>
    </el-form>

    <el-table :data="paginatedTasks" style="width: 100%">
      <el-table-column prop="applicationName" label="申请人"></el-table-column>
      <el-table-column prop="applicationSubject" label="申请主题"></el-table-column>
      <el-table-column prop="applicationDepartment" label="申请部门"></el-table-column>
      <el-table-column prop="applicationNumber" label="申请编号"></el-table-column>
      <el-table-column prop="applicationReason" label="申请原因"></el-table-column>
      <el-table-column prop="applicationType" label="申请类型"></el-table-column>
      <el-table-column prop="applicationVersion" label="申请类型"></el-table-column>
      <el-table-column prop="upgrade_package_name" label="升级包名称">
        <template slot-scope="scope">
          <a href="javascript:;" @click="downloadFile(scope.row.applicationName, scope.row.upgrade_package_name)">
            {{ scope.row.upgrade_package_name }}
          </a>
        </template>
      </el-table-column>

      <el-table-column prop="applicationTimes" label="申请时间"></el-table-column>

      <!-- 审核状态列，仅在展示历史信息时显示 -->
      <el-table-column prop="applicationStatus" label="审核状态" v-if="showHistory">
        <template v-slot:default="{ row }">
          {{ getStatusText(row.applicationStatus) }}
        </template>
      </el-table-column>

      <el-table-column v-if="!showHistory" label="操作">
        <template slot-scope="scope">
          <div class="action-buttons">
            <el-button
                size="mini"
                @click="approve(scope.row)"
                style="background-color: #67C23A; color: white; margin-bottom: 5px;"
            >
              审批通过
            </el-button>
          </div>
          <div class="action-buttons">
          <el-button
              size="mini"
              type="danger"
              @click="reject(scope.row)"
          >
            审批拒绝
          </el-button>
          </div>
        </template>
      </el-table-column>

    </el-table>

    <el-pagination
        @size-change="handleSizeChange"
        @current-change="handleCurrentChange"
        :current-page="currentPage"
        :page-sizes="[10, 20, 50, 100]"
        :page-size="pageSize"
        layout="total, sizes, prev, pager, next, jumper"
        :total="totalTasks">
    </el-pagination>
  </div>
  </div>
</template>

<script>
import IndexHeader from "@/components/index/IndexHeader.vue";
import base from "@/utils/base";

export default {
  components:{
    IndexHeader,
  },

  data() {
    return {
      searchForm: {
        applicationName: '',
        applicationDepartment: '',
        applicationNumber: '',
      },
      refreshInterval: null, // 用于保存定时器 ID
      refreshTime: 300000, // T 时间间隔（毫秒），定期刷新页面时间(5分钟刷新一次)
      tasks:[], // Initially generate 50 tasks for the demo
      // tasks: this.generateRandomTasks(50), // Initially generate 50 tasks for the demo
      filteredTasks: [],
      currentPage: 1,
      pageSize: 10,
      totalTasks: 0, // Total number of tasks after search
      showHistory:false,//展示历史信息
    };
  },
  computed: {
    paginatedTasks() {
      const start = (this.currentPage - 1) * this.pageSize;
      const end = start + this.pageSize;
      return this.filteredTasks.slice(start, end);
    }
  },
  mounted() {
    this.fetchTasks();
    this.refreshInterval = setInterval(() => {
      this.fetchTasks(); // 定时获取数据
    }, base.getRefreshTime());
    this.searchTasks();
  },
  beforeDestroy() {
    // 在组件销毁时清除定时器
    if (this.refreshInterval) {
      clearInterval(this.refreshInterval);
    }
  },

  methods: {
    toggleHistory() {
      this.showHistory = !this.showHistory;
      this.tasks = [];
      this.filteredTasks=[];
      this.showHistory ? this.fetchTaskHistory() : this.fetchTasks();
    },
    fetchTasks() {
      // 发起HTTP请求获取任务
      this.$http.get("/UpgradeTask/getAllUpgradeTask")
          .then(response => {
            const data = response.data;
            if (data && data.status === 0 && Array.isArray(data.data)) {
              // 用返回的任务列表更新tasks和filteredTasks
              this.tasks = data.data;
              console.log("Fetched tasks data:", JSON.stringify(this.tasks, null, 2));
              this.searchTasks(); // 进行搜索以更新filteredTasks和分页
            } else {
              // 处理错误或数据问题
              console.error("Failed to fetch tasks:", data.message);
              this.$message.error(data.message || "获取任务列表失败");
            }
          })
          .catch(error => {
            console.error("Error fetching tasks:", error);
            this.$message.error("获取任务列表时出错");
          });
    },
    fetchTaskHistory() {
      this.$http.get("/UpgradeTask/getUpgradeTaskHistory")
          .then(response => {
            const data = response.data;
            if (data && data.status === 0 && Array.isArray(data.data)) {
              // 用返回的任务列表更新tasks和filteredTasks
              this.tasks = data.data;

              // 添加排序步骤，假设 `applicationTimes` 是一个可以直接比较的日期字符串
              this.tasks.sort((a, b) => new Date(b.applicationTimes) - new Date(a.applicationTimes));

              this.searchTasks(); // 进行搜索以更新filteredTasks和分页
            } else {
              // 处理错误或数据问题
              console.error("Failed to fetch tasks:", data.message);
              this.$message.error(data.message || "获取历史任务列表失败");
            }
          })
          .catch(error => {
            console.error("Error fetching tasks:", error);
            this.$message.error("获取历史任务列表时出错");
          });
    },

    getStatusText(status) {
      const statusMap = {
        '0': '待审核',
        '1': '已通过',
        '2': '已拒绝',
      };
      return statusMap[status] || '未知';
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

    // generateRandomTasks(count) {
    //   const applicants = ["Alice", "Bob", "Charlie", "David"];
    //   const departments = ["Finance", "HR", "IT", "Marketing"];
    //   const tasks = [];
    //   for (let i = 0; i < count; i++) {
    //     tasks.push({
    //       applicant: applicants[Math.floor(Math.random() * applicants.length)],
    //       subject: "xxx",
    //       department: departments[Math.floor(Math.random() * departments.length)],
    //       applicationId: (Math.random().toString(36).substring(2, 5) + Math.random().toString(36).substring(2, 5)).toUpperCase(),
    //       reason: "xxx",
    //       upgradePackage:'xxx',
    //     });
    //   }
    //   return tasks;
    // },
    searchTasks() {
      this.filteredTasks = this.tasks.filter(task => {
        return (!this.searchForm.applicationName || task.applicationName.includes(this.searchForm.applicationName)) &&
            (!this.searchForm.applicationDepartment || task.applicationDepartment.includes(this.searchForm.applicationDepartment)) &&
            (!this.searchForm.applicationNumber || task.applicationNumber.includes(this.searchForm.applicationNumber));
      });
      this.totalTasks = this.filteredTasks.length;
      this.currentPage = 1; // Reset to first page after search
    },


    approve(row) {
      this.$confirm('请确认是否通过审批并下发升级任务', '提示', {
        confirmButtonText: '确认',
        cancelButtonText: '取消',
      }).then(() => {
        // 准备发送到后端的数据，确保它与后端期望的格式匹配
        const upgradeTaskDTO = {
          applicationNumber: row.applicationNumber, // 确保这里的属性与 DTO 一致
          applicationName: row.applicationName,
          applicationSubject: row.applicationSubject,
          applicationDepartment: row.applicationDepartment,
          applicationReason: row.applicationReason,
          applicationType: row.applicationType,
          upgrade_package_name: row.upgrade_package_name,
          applicationVersion: row.applicationVersion,
          applicationTimes: row.applicationTimes,
        };
        console.log("  Row: " + row.applicationNumber);
        // 发送 POST 请求
        this.$http.post('/UpgradeTask/agreeUpgradeTask', upgradeTaskDTO, {
          headers: {
            'Content-Type': 'application/json'
          },
        }).then(response => {
          if (response.data.status === 0) {
            this.$message({
              type: 'success',
              message: '升级任务已通过'
            });
            // 刷新任务列表或进行其他操作
            this.fetchTasks();
          } else {
            this.$message.error('操作失败: ' + response.data.message);
          }
        }).catch(error => {
          console.error("操作错误", error);
          this.$message.error('操作失败');
        });
      }).catch(() => {
        this.$message.info('操作取消');
      });
    },


    reject(row) {
      this.$prompt('请输入拒绝理由', '提示', {
        confirmButtonText: '确认',
        cancelButtonText: '取消',
        inputValidator: (value) => {
          return !!value || '理由不能为空';
        },
        inputErrorMessage: '理由不能为空'
      }).then(({ value }) => {
        // 准备发送到后端的数据，确保它与后端期望的格式匹配
        const upgradeTaskDTO = {
            applicationNumber: row.applicationNumber, // 确保这里的属性与 DTO 一致
            applicationName: row.applicationName,
            applicationSubject: row.applicationSubject,
            applicationDepartment: row.applicationDepartment,
            applicationReason: row.applicationReason,
            applicationType: row.applicationType,
            upgrade_package_name: row.upgrade_package_name,
            applicationTimes: row.applicationTimes,
        };
        console.log("ET:" + value + "  Row: " + row.applicationNumber);
        // 发送 POST 请求
        this.$http.post('/UpgradeTask/rejectUpgradeTask', upgradeTaskDTO, {
          headers: {
            'Content-Type': 'application/json'
          },
          params: {
            rejectReason: value,
          },
        }).then(response => {
          if (response.data.status === 0) {
            this.$message({
              type: 'success',
              message: '升级任务已拒绝'
            });
            // 刷新任务列表或进行其他操作
            this.fetchTasks();
          } else {
            this.$message.error('操作失败: ' + response.data.message);
          }
        }).catch(error => {
          console.error("操作错误", error);
          this.$message.error('操作失败');
        });
      }).catch(() => {
        this.$message.info('操作取消');
      });
    },



    handleSizeChange(newSize) {
      this.pageSize = newSize;
      this.handleCurrentChange(1); // Go back to first page after changing size
    },
    handleCurrentChange(newPage) {
      this.currentPage = newPage;
    }
  },

};
</script>

<style scoped>
.task-request-component {
  padding: 20px;
}
.search-form {
  margin-bottom: 20px;
}
</style>
