<template>
  <div class="main-content">
    <!-- 列表页 -->
    <div v-show="showFlag">
      <el-button type="primary" @click="isEditStockDialogVisible = true">新增库存</el-button>
      <el-dialog title="编辑库存" :visible.sync="isEditStockDialogVisible" width="30%">
        <el-form :model="editStockForm">
          <el-form-item label="托号">
            <el-input v-model="editStockForm.setNumber"></el-input>
          </el-form-item>
          <el-form-item label="状态">
            <el-input v-model="editStockForm.setStatus"></el-input>
          </el-form-item>
          <el-form-item label="原因">
            <el-input v-model="editStockForm.setReason"></el-input>
          </el-form-item>
          <el-form-item label="模组电池型号">
            <el-input v-model="editStockForm.add_time"></el-input>
          </el-form-item>
          <el-form-item>
            <el-button type="primary" @click="handleEditStock">确认修改</el-button>
            <el-button @click="isEditStockDialogVisible = false">取消</el-button>
          </el-form-item>
        </el-form>
      </el-dialog>

      <el-dialog title="添加新库存" :visible.sync="isAddStockDialogVisible" width="30%">
        <el-form :model="newStockForm">
          <el-form-item label="托号">
            <el-input v-model="newStockForm.setNumber"></el-input>
          </el-form-item>
          <el-form-item label="状态">
            <el-input v-model="newStockForm.setStatus"></el-input>
          </el-form-item>
          <el-form-item label="分区">
            <el-input v-model="newStockForm.setReason"></el-input>
          </el-form-item>
          <el-form-item label="模组电池型号">
            <el-input v-model="newStockForm.add_time"></el-input>
          </el-form-item>
          <el-form-item>
            <el-button type="primary" @click="addStock">确认添加</el-button>
            <el-button @click="isAddStockDialogVisible = false">取消</el-button>
          </el-form-item>
        </el-form>
      </el-dialog>

      <div class="table-content">
        <el-table :data="stocks" :stripe="true">
          <el-table-column prop="setNumber" label="托号"></el-table-column>
          <el-table-column prop="setStatus" label="状态"></el-table-column>
          <el-table-column prop="setReason" label="原因"></el-table-column>
          <el-table-column prop="add_time" label="添加时间">
            <template slot-scope="scope">
              {{ formatDate(scope.row.add_time) }}
            </template>
          </el-table-column>
          <el-table-column label="操作">
            <template slot-scope="scope">
<!--              <el-button type="primary" @click="openEditDialog(scope.row)">编辑</el-button>-->
              <el-button type="danger" @click="deleteStock(scope.row)">删除</el-button>
            </template>
          </el-table-column>
        </el-table>
      </div>
    </div>
  </div>
</template>


<script>
import moment from 'moment';

export default {
  data() {
    return {
      showFlag: true,
      isEditStockDialogVisible: false,
      isAddStockDialogVisible: false,
      stocks: [],
      newStockForm: {
        setNumber: '',
        setStatus: '',
        setReason: '',
        add_time: ''
      },
      editStockForm: {}
    };
  },
  methods: {
    formatDate(date) {
      return moment(date).format('YYYY-MM-DD HH:mm:ss');
    },

    getModuleCategoryLabel(moduleCategory) {
      const labels = {
        '1': 'NCM_140A',
        '2': 'NCM_200A',
        '3': 'LFP'
      };
      return labels[moduleCategory] || 'Unknown'; // Returns 'Unknown' if no match is found
    },

    openEditDialog(stock) {
      this.editStockForm = Object.assign({}, stock);
      this.isEditStockDialogVisible = true;
    },
    handleEditStock() {
      this.$http.post('/Unchecked/editStock', this.editStockForm).then(response => {
        if (response.data.status === 'success') {
          this.$message.success('库存更新成功');
          this.isEditStockDialogVisible = false;
          this.fetchStocks();
        } else {
          this.$message.error('库存更新失败: ' + response.data.message);
        }
      }).catch(error => {
        this.$message.error('网络错误或服务器异常: ' + error.message);
      });
    },
    addStock() {
      this.$http.post('/Unchecked/addStock', this.newStockForm).then(response => {
        if (response.data.status === 'success') {
          this.$message.success('库存添加成功');
          this.isAddStockDialogVisible = false;
          this.fetchStocks();
        } else {
          this.$message.error('库存添加失败: ' + response.data.message);
        }
      }).catch(error => {
        this.$message.error('网络错误或服务器异常');
      });
    },
    deleteStock(stock) {
      this.$confirm('确定要删除这个库存吗?', '警告', {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning'
      }).then(() => {
        this.$http.post('/Unchecked/deleteStock', { setNumber: stock.setNumber }).then(response => {
          if (response.data.status === 'success') {
            this.$message.success('库存删除成功');
            this.fetchStocks();
          } else {
            this.$message.error('库存删除失败: ' + response.data.message);
          }
        }).catch(error => {
          this.$message.error('删除失败: ' + error.message);
        });
      }).catch(() => {
        this.$message.info('删除操作已取消');
      });
    },

    fetchStocks() {
      this.$http.get('/Unchecked/infos').then(response => {
        if (response.data.status === 0) {
          this.stocks = response.data.data;
        } else {
          this.$message.error('获取库存数据失败');
        }
      }).catch(error => {
        this.$message.error('网络错误或服务器异常: ' + error.message);
      });
    }
  },
  mounted() {
    this.fetchStocks();
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
