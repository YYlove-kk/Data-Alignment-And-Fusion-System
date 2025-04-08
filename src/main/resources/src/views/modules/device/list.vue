<template>
  <div class="main-content">
    <!-- 列表页 -->
    <div v-show="showFlag">
      <el-button type="primary" @click="isEditDeviceDialogVisible = true">新增设备</el-button>
      <el-dialog title="编辑设备" :visible.sync="isEditDeviceDialogVisible" width="30%">
        <el-form :model="editDeviceForm">
          <el-form-item label="设备名称">
            <el-input v-model="editDeviceForm.deviceName" disabled></el-input>
          </el-form-item>
          <el-form-item label="设备启动星期">
            <el-select v-model="editDeviceForm.deviceStartDay" placeholder="请选择">
              <el-option v-for="day in daysOfWeek" :key="day.value" :label="day.label" :value="day.value"></el-option>
            </el-select>
          </el-form-item>
          <el-form-item label="设备启动小时">
            <el-select v-model="editDeviceForm.deviceStartHour" placeholder="请选择">
              <el-option v-for="hour in hoursOfDay" :key="hour" :label="hour + '时'" :value="hour"></el-option>
            </el-select>
          </el-form-item>
          <el-form-item label="设备启动分钟">
            <el-select v-model="editDeviceForm.deviceStartMinute" placeholder="请选择">
              <el-option v-for="minute in minutesOfHour" :key="minute" :label="minute + '分'" :value="minute"></el-option>
            </el-select>
          </el-form-item>
          <el-form-item>
            <el-button type="primary" @click="handleEditDevice">确认修改</el-button>
            <el-button @click="isEditDeviceDialogVisible = false">取消</el-button>
          </el-form-item>
        </el-form>
      </el-dialog>

      <el-dialog title="添加新设备" :visible.sync="isAddDeviceDialogVisible" width="30%">
        <el-form :model="newDeviceForm">
          <el-form-item label="设备名称">
            <el-input v-model="newDeviceForm.deviceName"></el-input>
          </el-form-item>
          <el-form-item label="设备启动星期">
            <el-select v-model="newDeviceForm.deviceStartDay" placeholder="请选择">
              <el-option v-for="day in daysOfWeek" :key="day.value" :label="day.label" :value="day.value"></el-option>
            </el-select>
          </el-form-item>
          <el-form-item label="设备启动小时">
            <el-select v-model="newDeviceForm.deviceStartHour" placeholder="请选择">
              <el-option v-for="hour in hoursOfDay" :key="hour" :label="hour + '时'" :value="hour"></el-option>
            </el-select>
          </el-form-item>
          <el-form-item label="设备启动分钟">
            <el-select v-model="newDeviceForm.deviceStartMinute" placeholder="请选择">
              <el-option v-for="minute in minutesOfHour" :key="minute" :label="minute + '分'" :value="minute"></el-option>
            </el-select>
          </el-form-item>
          <el-form-item>
            <el-button type="primary" @click="addDevice">确认添加</el-button>
            <el-button @click="isAddDeviceDialogVisible = false">取消</el-button>
          </el-form-item>
        </el-form>
      </el-dialog>

      <div class="table-content">
        <el-table class="tables" :size="contents.tableSize" :show-header="contents.tableShowHeader"
            :header-row-style="headerRowStyle" :header-cell-style="headerCellStyle"
            :border="contents.tableBorder"
            :fit="contents.tableFit"
            :stripe="contents.tableStripe"
            :row-style="rowStyle"
            :cell-style="cellStyle"
            :style="{width: '100%',fontSize:contents.tableContentFontSize,color:contents.tableContentFontColor}"
            :data="paginatedDataList"
            v-loading="dataListLoading"
            @selection-change="selectionChangeHandler">


            <el-table-column  v-if="contents.tableSelection"
                type="selection"
                header-align="center"
                align="center"
                width="50">
            </el-table-column>

          <el-table-column prop="id" label="ID" width="120"></el-table-column>
          <el-table-column prop="deviceName" label="设备名称"></el-table-column>
          <el-table-column
              prop="deviceStartDay"
              label="设备启动星期"
              :formatter="formatDayOfWeek">
          </el-table-column>
          <el-table-column prop="deviceStartHour" label="设备启动小时"></el-table-column>
          <el-table-column prop="deviceStartMinute" label="设备启动分钟"></el-table-column>
          <el-table-column prop="deviceModuleAccount" label="设备监控模组数量"></el-table-column>
          <el-table-column prop="deviceStatus" label="设备状态"></el-table-column>
          <el-table-column prop="deviceAddtime" label="添加时间"></el-table-column>
          <el-table-column label="操作" fixed="right">

        <template slot-scope="scope">
          <div>
            <el-button
                type="primary"
                icon="el-icon-edit"
                size="mini"
                @click="openEditDialog(scope.row)">
              编辑
            </el-button>
          </div>
          <div>
            <el-button type="danger" icon="el-icon-delete" size="mini" @click="deleteHandler(scope.row)">
              删除
            </el-button>
          </div>

        </template>

        </el-table-column>

        </el-table>
        <el-pagination
          clsss="pages"
          :layout="layouts"
          @size-change="sizeChangeHandle"
          @current-change="currentChangeHandle"
          :current-page="pageIndex"
          :page-sizes="[10, 20, 50, 100]"
          :page-size="Number(contents.pageEachNum)"
          :total="totalPage"
          :small="contents.pageStyle"
          class="pagination-content"
          :background="contents.pageBtnBG"
          :style="{textAlign:contents.pagePosition==1?'left':contents.pagePosition==2?'center':'right'}"
        ></el-pagination>
      </div>
    </div>
    <!-- 添加/修改页面  将父组件的search方法传递给子组件-->
      <detail v-show="showDetail" :parent="this" :detailData="currentRowData" ref="detail"></detail>
    </div>
</template>
<script>

import detail from "@/views/modules/device/detail.vue";
import * as XLSX from 'xlsx';
import base from "@/utils/base";
export default {
  props: ['deviceName', 'deviceArea','deviceMemory','deviceSoc','deviceLoadBytes','deviceStatus','deviceNoduleAccount','deviceAddtime'],
  data() {
    return {
      searchForm: {
        device_name: "",
        device_area: "",
        device_memory: "",
        // 其他字段保持不变
      },
      refreshInterval: null, // 用于保存定时器 ID
      refreshTime: 300000, // T 时间间隔（毫秒），定期刷新页面时间(5分钟刷新一次)
      fullDataList:[],
      dataList:[],
      paginatedDataList:[],
      currentRowData: {},
      pageIndex: 1,
      pageSize: 10,
      totalPage: 0,
      dataListLoading: false,
      dataListSelections: [],
      showFlag: true,
      showDetail:false,
      contents:{"searchBtnFontColor":"rgba(0, 0, 0, 1)","pagePosition":"1","inputFontSize":"14px","inputBorderRadius":"0px","tableBtnDelFontColor":"#333","tableBtnIconPosition":"1","searchBtnHeight":"40px","inputIconColor":"rgba(0, 0, 0, 1)","searchBtnBorderRadius":"4px","tableStripe":true,"btnAdAllWarnFontColor":"#333","tableBtnDelBgColor":"rgba(204, 255, 102, 1)","searchBtnIcon":"1","tableSize":"medium","searchBtnBorderStyle":"groove","tableSelection":true,"searchBtnBorderWidth":"1px","tableContentFontSize":"14px","searchBtnBgColor":"#fff","inputTitleSize":"14px","btnAdAllBorderColor":"#DCDFE6","pageJumper":true,"btnAdAllIconPosition":"1","searchBoxPosition":"1","tableBtnDetailFontColor":"#333","tableBtnHeight":"40px","pagePager":true,"searchBtnBorderColor":"rgba(130, 131, 133, 1)","tableHeaderFontColor":"#909399","inputTitle":"1","tableBtnBorderRadius":"4px","btnAdAllFont":"0","btnAdAllDelFontColor":"#333","tableBtnIcon":"1","btnAdAllHeight":"40px","btnAdAllWarnBgColor":"rgba(204, 255, 102, 1)","btnAdAllBorderWidth":"1px","tableStripeFontColor":"#606266","tableBtnBorderStyle":"solid","inputHeight":"40px","btnAdAllBorderRadius":"4px","btnAdAllDelBgColor":"rgba(102, 204, 255, 1)","pagePrevNext":true,"btnAdAllAddBgColor":"rgba(117, 113, 249, 1)","searchBtnFont":"1","tableIndex":true,"btnAdAllIcon":"1","tableSortable":false,"pageSizes":true,"tableFit":true,"pageBtnBG":true,"searchBtnFontSize":"15px","tableBtnEditBgColor":"rgba(102, 204, 255, 1)","inputBorderWidth":"2px","inputFontPosition":"1","inputFontColor":"rgba(0, 0, 0, 1)","pageEachNum":10,"tableHeaderBgColor":"#fff","inputTitleColor":"#333","btnAdAllBoxPosition":"3","tableBtnDetailBgColor":"rgba(117, 113, 249, 1)","inputIcon":"1","searchBtnIconPosition":"1","btnAdAllFontSize":"14px","inputBorderStyle":"none none solid none ","inputBgColor":"rgba(117, 113, 249, 0.09)","pageStyle":false,"pageTotal":true,"btnAdAllAddFontColor":"#333","tableBtnFont":"1","tableContentFontColor":"#606266","inputBorderColor":"rgba(115, 115, 115, 1)","tableShowHeader":true,"tableBtnFontSize":"14px","tableBtnBorderColor":"#DCDFE6","inputIconPosition":"2","tableBorder":true,"btnAdAllBorderStyle":"solid","tableBtnBorderWidth":"1px","tableStripeBgColor":"#F5F7FA","tableBtnEditFontColor":"#333","tableAlign":"center"},
      layouts: '',

      // 现有的数据保持不变
      isAddDeviceDialogVisible: false,
      newDeviceForm: {
        deviceName: '',
        deviceStartDay: '',
        deviceStartHour: '',
        deviceStartMinute: '',
      },
      daysOfWeek: [
        { value: 0, label: '星期一' },
        { value: 1, label: '星期二' },
        { value: 2, label: '星期三' },
        { value: 3, label: '星期四' },
        { value: 4, label: '星期五' },
        { value: 5, label: '星期六' },
        { value: 6, label: '星期日' }
      ],
      hoursOfDay: Array.from({ length: 24 }, (_, i) => i),
      minutesOfHour: Array.from({ length: 60 }, (_, i) => i),
      isEditDeviceDialogVisible: false,
      editDeviceForm: {
        deviceName: '',
        deviceStartDay: '',
        deviceStartHour: '',
        deviceStartMinute: '',
      },
    };
  },

  watch: {
    pageSize(newVal) {
      this.pageSize = newVal;
      const startIndex = (this.pageIndex - 1) * this.pageSize;
      const endIndex = startIndex + this.pageSize;
      this.paginatedDataList = this.dataList.slice(startIndex, endIndex);
    },
    pageIndex(newVal) {
      this.pageIndex = newVal;
      const startIndex = (this.pageIndex - 1) * this.pageSize;
      const endIndex = startIndex + this.pageSize;
      this.paginatedDataList = this.dataList.slice(startIndex, endIndex);
    },
    showFlag(newVal){
      this.init();
      this.generateRandomData();
      //TODO 链接服务器后需要据此修改
      // this.getFullDataList();
      this.dataList = this.fullDataList;
      // this.getDataList();
      //第一次加载时搜索所有数据并展示出来
      this.search();
    }
  },

  created() {
    this.init();
  },

  mounted() {
    this.getFullDataList();
    this.refreshInterval = setInterval(() => {
      this.getFullDataList(); // 定时获取数据
    }, base.getRefreshTime()); // 使用base.js中的refreshTime
    this.contentStyleChange();
  },
  beforeDestroy() {
    // 在组件销毁时清除定时器
    if (this.refreshInterval) {
      clearInterval(this.refreshInterval);
    }
  },
  filters: {
    htmlfilter: function (val) {
      return val.replace(/<[^>]*>/g).replace(/undefined/g,'');
    }
  },
  components: {
    detail,
  },
  methods: {
    // 初始化
    init () {
    },

    openEditDialog(device) {
      // 使用对象展开操作符来复制设备对象到表单数据
      this.editDeviceForm = { ...device };
      this.isEditDeviceDialogVisible = true;
    },

    closeEditDialog() {
      this.isEditDeviceDialogVisible = false;
    },
    handleEditDevice() {
      this.$http.post('/Device/editDevice', this.editDeviceForm).then(response => {
        if (response.data.status === 0) {
          this.$message.success('设备修改成功');
          this.isEditDeviceDialogVisible = false;
          this.getFullDataList(); // 刷新设备列表
        } else {
          this.$message.error('设备修改失败: ' + response.data.message);
        }
      }).catch(error => {
        this.$message.error('网络错误或服务器异常: ' + error.message);
      });
    },

    formatDayOfWeek(row, column, value) {
      const day = this.daysOfWeek.find(day => day.value === value);
      return day ? day.label : '未知';
    },

    addDevice() {
      const deviceDTO = {
        deviceName: this.newDeviceForm.deviceName,
        deviceStatus: "重启", // 根据业务逻辑调整
        deviceModuleAccount: 0, // 默认模组数量，根据业务逻辑调整
        deviceStartDay: parseInt(this.newDeviceForm.deviceStartDay), // 确保为整数
        deviceStartHour: parseInt(this.newDeviceForm.deviceStartHour), // 确保为整数
        deviceStartMinute: parseInt(this.newDeviceForm.deviceStartMinute), // 确保为整数
        deviceAddtime: this.formatDate(new Date()) // 使用格式化函数设置当前时间
      };

      this.$http.post('/Device/addDevice', deviceDTO).then(response => {
        if (response.data.status === 0) {
          this.$message.success('设备添加成功');
          this.isAddDeviceDialogVisible = false;
          this.getFullDataList(); // 刷新设备列表
        } else {
          this.$message.error('设备添加失败');
        }
      }).catch(error => {
        this.$message.error('网络错误或服务器异常');
      });
    },

    formatDate(date) {
      const options = { year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false };
      return new Intl.DateTimeFormat('zh-CN', options).format(date).replace(/\//g, '-').replace(',', '');
    },

    //查询功能
    search() {
      this.dataList = this.fullDataList;
      // 使用filter方法根据搜索条件过滤fullDataList
      this.dataList = this.fullDataList.filter((item) => {
        return (!this.searchForm.device_name || item.device_name === this.searchForm.device_name) &&
            (!this.searchForm.device_area || item.device_area === this.searchForm.device_area) &&
            (!this.searchForm.device_memory || item.device_memory.toUpperCase() === this.searchForm.device_memory.toUpperCase());
      });
      // 更新分页数据
      this.totalPage = this.dataList.length;
      this.pageIndex = 1; // 重置到第一页
      const startIndex = (this.pageIndex - 1) * this.pageSize;
      const endIndex = startIndex + this.pageSize;
      this.paginatedDataList = this.dataList.slice(startIndex, endIndex);
      this.initPaginatedDataList();
    },

    // 获取所有数据列表
    async getFullDataList() {
      this.dataListLoading = true;

      try {
        const response = await this.$http.get("Device/infos");
        const data = response.data;

        if (data && data.status === 0) {
          this.fullDataList = data.data;
          // 将fullDataList的数据赋值给dataList以供操作，确保原始数据不被修改
          this.dataList = [...this.fullDataList];
          this.totalPage = data.total || data.data.length;
          // 初始化paginatedDataList
          this.search();
        } else {
          this.fullDataList = [];
          this.dataList = [];
          this.totalPage = 0;
        }
      } catch (error) {
        console.error("Failed to fetch device data:", error);
      } finally {
        this.dataListLoading = false;
      }
    },

    initPaginatedDataList(){
      this.pageSize = 10;
      this.pageIndex = 1;
      this.totalPage = this.dataList.length;
      const startIndex = (this.pageIndex - 1) * this.pageSize;
      const endIndex = startIndex + this.pageSize;
      this.paginatedDataList = this.dataList.slice(startIndex, endIndex);
    },
    contentStyleChange() {
      this.contentSearchStyleChange()
      this.contentBtnAdAllStyleChange()
      this.contentSearchBtnStyleChange()
      this.contentTableBtnStyleChange()
      this.contentPageStyleChange()
    },
    contentSearchStyleChange() {
      this.$nextTick(()=>{
        document.querySelectorAll('.form-content .slt .el-input__inner').forEach(el=>{
          let textAlign = 'left'
          if(this.contents.inputFontPosition == 2) textAlign = 'center'
          if(this.contents.inputFontPosition == 3) textAlign = 'right'
          el.style.textAlign = textAlign
          el.style.height = this.contents.inputHeight
          el.style.lineHeight = this.contents.inputHeight
          el.style.color = this.contents.inputFontColor
          el.style.fontSize = this.contents.inputFontSize
          el.style.borderWidth = this.contents.inputBorderWidth
          el.style.borderStyle = this.contents.inputBorderStyle
          el.style.borderColor = this.contents.inputBorderColor
          el.style.borderRadius = this.contents.inputBorderRadius
          el.style.backgroundColor = this.contents.inputBgColor
        })
        if(this.contents.inputTitle) {
          document.querySelectorAll('.form-content .slt .el-form-item__label').forEach(el=>{
            el.style.color = this.contents.inputTitleColor
            el.style.fontSize = this.contents.inputTitleSize
            el.style.lineHeight = this.contents.inputHeight
          })
        }
        setTimeout(()=>{
          document.querySelectorAll('.form-content .slt .el-input__prefix').forEach(el=>{
            el.style.color = this.contents.inputIconColor
            el.style.lineHeight = this.contents.inputHeight
          })
          document.querySelectorAll('.form-content .slt .el-input__suffix').forEach(el=>{
            el.style.color = this.contents.inputIconColor
            el.style.lineHeight = this.contents.inputHeight
          })
          document.querySelectorAll('.form-content .slt .el-input__icon').forEach(el=>{
            el.style.lineHeight = this.contents.inputHeight
          })
        },10)

      })
    },
    // 搜索按钮
    contentSearchBtnStyleChange() {
      this.$nextTick(()=>{
        document.querySelectorAll('.form-content .slt .el-button--success').forEach(el=>{
          el.style.height = this.contents.searchBtnHeight
          el.style.color = this.contents.searchBtnFontColor
          el.style.fontSize = this.contents.searchBtnFontSize
          el.style.borderWidth = this.contents.searchBtnBorderWidth
          el.style.borderStyle = this.contents.searchBtnBorderStyle
          el.style.borderColor = this.contents.searchBtnBorderColor
          el.style.borderRadius = this.contents.searchBtnBorderRadius
          el.style.backgroundColor = this.contents.searchBtnBgColor
        })
      })
    },
    // 新增、批量删除
    contentBtnAdAllStyleChange() {
      this.$nextTick(()=>{
        document.querySelectorAll('.form-content .ad .el-button--success').forEach(el=>{
          el.style.height = this.contents.btnAdAllHeight
          el.style.color = this.contents.btnAdAllAddFontColor
          el.style.fontSize = this.contents.btnAdAllFontSize
          el.style.borderWidth = this.contents.btnAdAllBorderWidth
          el.style.borderStyle = this.contents.btnAdAllBorderStyle
          el.style.borderColor = this.contents.btnAdAllBorderColor
          el.style.borderRadius = this.contents.btnAdAllBorderRadius
          el.style.backgroundColor = this.contents.btnAdAllAddBgColor
        })
        document.querySelectorAll('.form-content .ad .el-button--danger').forEach(el=>{
          el.style.height = this.contents.btnAdAllHeight
          el.style.color = this.contents.btnAdAllDelFontColor
          el.style.fontSize = this.contents.btnAdAllFontSize
          el.style.borderWidth = this.contents.btnAdAllBorderWidth
          el.style.borderStyle = this.contents.btnAdAllBorderStyle
          el.style.borderColor = this.contents.btnAdAllBorderColor
          el.style.borderRadius = this.contents.btnAdAllBorderRadius
          el.style.backgroundColor = this.contents.btnAdAllDelBgColor
        })
        document.querySelectorAll('.form-content .ad .el-button--warning').forEach(el=>{
          el.style.height = this.contents.btnAdAllHeight
          el.style.color = this.contents.btnAdAllWarnFontColor
          el.style.fontSize = this.contents.btnAdAllFontSize
          el.style.borderWidth = this.contents.btnAdAllBorderWidth
          el.style.borderStyle = this.contents.btnAdAllBorderStyle
          el.style.borderColor = this.contents.btnAdAllBorderColor
          el.style.borderRadius = this.contents.btnAdAllBorderRadius
          el.style.backgroundColor = this.contents.btnAdAllWarnBgColor
        })
      })
    },
    // 表格
    rowStyle({ row, rowIndex}) {
      if (rowIndex % 2 == 1) {
        if(this.contents.tableStripe) {
          return {color:this.contents.tableStripeFontColor}
        }
      } else {
        return ''
      }
    },
    cellStyle({ row, rowIndex}){
      if (rowIndex % 2 == 1) {
        if(this.contents.tableStripe) {
          return {backgroundColor:this.contents.tableStripeBgColor}
        }
      } else {
        return ''
      }
    },
    headerRowStyle({ row, rowIndex}){
      return {color: this.contents.tableHeaderFontColor}
    },
    headerCellStyle({ row, rowIndex}){
      return {backgroundColor: this.contents.tableHeaderBgColor}
    },
    // 表格按钮
    contentTableBtnStyleChange(){
    },
    // 分页
    contentPageStyleChange(){
      let arr = []

      if(this.contents.pageTotal) arr.push('total')
      if(this.contents.pageSizes) arr.push('sizes')
      if(this.contents.pagePrevNext){
        arr.push('prev')
        if(this.contents.pagePager) arr.push('pager')
        arr.push('next')
      }
      if(this.contents.pageJumper) arr.push('jumper')
      this.layouts = arr.join()
      this.contents.pageEachNum = 10
    },
    // 每页数
    sizeChangeHandle(val) {
      this.pageSize = val;
      this.pageIndex = 1;
    },
    // 当前页
    currentChangeHandle(val) {
      this.pageIndex = val;
    },

    // 确保当详情页返回列表页时，更新数据列表和分页设置
    backToListFromDetail() {
      this.showDetail = false;
      this.showFlag = true;
    },

    // 多选
    selectionChangeHandler(val) {
      this.dataListSelections = val;
    },

    // 展示详情信息
    showDetailHandler(rowData) {
      // console.log("RD: " + rowData.device_status);
      this.showDetail = true;
      this.showFlag = false;
      // 将当前行的数据存储为组件的属性
      this.currentRowData = rowData;
    },

    //TODO 上传设备信息
    deviceUpload(file) {
      const isExcel = file.type === 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' || file.type === 'application/vnd.ms-excel';
      if (!isExcel) {
        this.$message.error('文件格式错误，请上传Excel文件！');
        return false;
      }
      return true;
    },



    // 删除
    deleteHandler(device) {
      this.$confirm('确定要删除这个设备吗?', '警告', {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning'
      }).then(() => {
        // Creating a deviceDTO object from the device data
        const deviceDTO = {
          id: device.id,
          deviceName: device.deviceName,
          deviceArea: device.deviceArea,
          deviceMemory: device.deviceMemory,
          deviceSoc: device.deviceSoc,
          deviceLoadBytes: device.deviceLoadBytes,
          deviceStatus: device.deviceStatus,
          deviceModuleAccount: device.deviceModuleAccount,
          deviceAddtime: device.deviceAddtime
        };

        // Sending the deviceDTO object as part of the POST request
        this.$http.post('/Device/deleteDevice', deviceDTO)
            .then(response => {
              if (response.data.status === 0) {
                this.$message.success('设备删除成功');
                this.getFullDataList(); // Refresh the list to show updated data
              } else {
                this.$message.error('设备删除失败: ' + response.data.message);
              }
            })
            .catch(error => {
              this.$message.error('删除失败: ' + error.message);
            });
      }).catch(() => {
        this.$message.info('删除操作已取消');
      });
    }


  },

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
