<template>
  <div class="main-content">
    <!-- 新增分区按钮和对话框 -->
    <el-button type="primary" @click="showAddDialog">新增分区</el-button>
    <el-dialog title="添加新分区" :visible.sync="isAddRegionDialogVisible" width="30%">
      <el-form :model="newRegionForm">
        <el-form-item label="分区名称">
          <el-input v-model="newRegionForm.regionName"></el-input>
        </el-form-item>
        <el-form-item label="设备名称">
          <el-input v-model="newRegionForm.regionDevice"></el-input>
        </el-form-item>
        <el-form-item label="分区状态">
          <el-input v-model="newRegionForm.regionStatus"></el-input>
        </el-form-item>
        <el-form-item label="分区描述">
          <el-input v-model="newRegionForm.regionDescribe"></el-input>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="handleAddRegion">确认添加</el-button>
          <el-button @click="closeAddDialog">取消</el-button>
        </el-form-item>
      </el-form>
    </el-dialog>

    <!-- 编辑分区对话框 -->
    <el-dialog title="编辑分区" :visible.sync="isEditRegionDialogVisible" width="30%">
      <el-form :model="editRegionForm">
        <el-form-item label="分区名称">
          <el-input v-model="editRegionForm.regionName" disabled></el-input>
        </el-form-item>
        <el-form-item label="设备名称">
          <el-input v-model="editRegionForm.regionDevice"></el-input>
        </el-form-item>
        <el-form-item label="分区状态">
          <el-input v-model="editRegionForm.regionStatus"></el-input>
        </el-form-item>
        <el-form-item label="分区描述">
          <el-input v-model="editRegionForm.regionDescribe"></el-input>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="handleEditRegion">确认</el-button>
          <el-button @click="isEditRegionDialogVisible = false">取消</el-button>
        </el-form-item>
      </el-form>
    </el-dialog>

    <!-- 分区列表 -->
    <div class="table-content">
      <el-table :data="regionsList" v-loading="loading" style="width: 100%">
        <el-table-column type="selection" width="50"></el-table-column>
        <el-table-column prop="regionName" label="分区名称"></el-table-column>
        <el-table-column prop="regionDevice" label="设备信息"></el-table-column>
        <el-table-column prop="regionStatus" label="分区状态"></el-table-column>
        <el-table-column prop="regionDescribe" label="描述信息"></el-table-column>
        <el-table-column label="操作">
          <template slot-scope="scope">
            <el-button type="primary" size="mini" @click="openEditDialog(scope.row)">编辑</el-button>
            <el-button type="danger" size="mini" @click="confirmDeleteRegion(scope.row)">删除</el-button>
          </template>
        </el-table-column>
      </el-table>
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      regionsList: [],
      loading: false,
      isAddRegionDialogVisible: false,
      isEditRegionDialogVisible: false,
      newRegionForm: {
        regionName: '',
        regionDevice: '',
        regionStatus: '',
        regionDescribe: ''
      },
      editRegionForm: {}
    };
  },
  methods: {
    fetchAllRegions() {
      this.loading = true;
      this.$http.get('/Region/infos').then(response => {
        this.loading = false;
        if (response.data.status === 0) {
          this.regionsList = response.data.data;
        } else {
          this.$message.error('分区信息加载失败: ' + response.data.message);
        }
      }).catch(error => {
        this.loading = false;
        this.$message.error('网络错误或服务器异常: ' + error.message);
      });
    },
    showAddDialog() {
      this.isAddRegionDialogVisible = true;
    },
    closeAddDialog() {
      this.isAddRegionDialogVisible = false;
    },
    handleAddRegion() {
      this.$http.post('/Region/addRegion', this.newRegionForm).then(response => {
        if (response.data.status === 0) {
          this.$message.success(response.data.message);
          this.closeAddDialog();
          this.fetchAllRegions();
        } else {
          this.$message.error('分区添加失败: ' + response.data.message);
        }
      }).catch(error => {
        this.$message.error('添加分区时网络错误或服务器异常: ' + error.message);
      });
    },
    openEditDialog(region) {
      this.editRegionForm = { ...region };
      this.isEditRegionDialogVisible = true;
    },
    handleEditRegion() {
      this.$http.post('/Region/updateRegion', this.editRegionForm).then(response => {
        if (response.data.status === 0) {
          this.$message.success(response.data.message);
          this.isEditRegionDialogVisible = false;
          this.fetchAllRegions();
        } else {
          this.$message.error('分区更新失败: ' + response.data.message);
        }
      }).catch(error => {
        this.$message.error('更新分区时网络错误或服务器异常: ' + error.message);
      });
    },
    // 删除分区前确认
    confirmDeleteRegion(region) {
      this.$confirm('是否删除分区？', '警告', {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning'
      }).then(() => {
        this.deleteRegion(region);
      }).catch(() => {
        this.$message.info('已取消删除');
      });
    },

    deleteRegion(region) {
      this.$http.post('/Region/deleteRegion', { regionName: region.regionName }).then(response => {
        if (response.data.status === 0) {
          this.$message.success('分区删除成功');
          this.fetchAllRegions();
        } else {
          this.$message.error('分区删除失败: ' + response.data.message);
        }
      }).catch(error => {
        this.$message.error('删除分区时网络错误或服务器异常: ' + error.message);
      });
    }
  },
  mounted() {
    this.fetchAllRegions();
  }
};
</script>

<style lang="scss" scoped>
.slt {
  margin: 0 !important;
  display: flex;
}

.ad {
  margin: 0 !important;
  display: flex;
}

.pages {
  & /deep/ el-pagination__sizes{
    & /deep/ el-input__inner {
      height: 22px;
      line-height: 22px;
    }
  }
}

.tables {
  & /deep/ .el-button--success {
    height: 40px;
    color: #333;
    font-size: 14px;
    border-width: 1px;
    border-style: solid;
    border-color: #DCDFE6;
    border-radius: 4px;
    background-color: rgba(117, 113, 249, 1);
  }

  & /deep/ .el-button--primary {
    height: 40px;
    color: #333;
    font-size: 14px;
    border-width: 1px;
    border-style: solid;
    border-color: #DCDFE6;
    border-radius: 4px;
    background-color: rgba(102, 204, 255, 1);
  }

  & /deep/ .el-button--danger {
    height: 40px;
    color: #333;
    font-size: 14px;
    border-width: 1px;
    border-style: solid;
    border-color: #DCDFE6;
    border-radius: 4px;
    background-color: rgba(204, 255, 102, 1);
  }
}

@media (max-width: 768px) { // 针对平板和手机屏幕调整样式
  .slt, .ad {
    flex-direction: column; // 在小屏幕上改为垂直布局
  }

  .tables & /deep/ .el-table {
    & /deep/ .el-table__body,
    & /deep/ .el-table__header {
      font-size: 14px; // 减小表格字体，适应小屏幕
    }

    & /deep/ .el-table__cell,
    & /deep/ .el-table-column {
      padding: 8px; // 减少单元格内边距以节省空间
    }

    & /deep/ .el-table-column {
      min-width: 80px; // 设置最小宽度保证内容可见
    }
  }

  .pages {
    & /deep/ .el-pagination {
      flex-wrap: wrap; // 分页组件在小屏幕上允许内容换行
      justify-content: space-between; // 保证元素间隔均匀
    }

    & /deep/ .el-pagination__total,
    & /deep/ .el-pagination__sizes,
    & /deep/ .el-pagination__pager {
      margin-bottom: 10px; // 增加底部外边距，避免元素挤压
    }
  }
}

</style>
