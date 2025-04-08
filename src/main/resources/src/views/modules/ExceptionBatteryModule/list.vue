<template>
  <div class="main-content">
    <!-- 列表页 -->
    <div v-show="showFlag">


      <div class="exception-status-buttons">
        <el-button
            v-for="status in exceptionStatuses"
            :key="status"
            :type="selectedExceptionStatus === status ? 'warning' : 'primary'"
            @click="updateExceptionStatus(status)"
            class="exception-btn">
          {{ status }}
        </el-button>
      </div>

<!--      <div class="exception-status-buttons" style="display: flex; justify-content: start; gap: 20px; padding-left: 0%; padding-right: 0%; height: 60px">-->
<!--        <el-button-->
<!--            v-for="status in exceptionStatuses"-->
<!--            :key="status"-->
<!--            :type="selectedExceptionStatus === status ? 'warning' : 'primary'"-->
<!--            @click="updateExceptionStatus(status)"-->
<!--            :style="buttonStyle(status)">-->
<!--          {{ status }}-->
<!--        </el-button>-->
<!--      </div>-->

      <el-form :inline="true" :model="searchForm" class="form-content">
        <el-row :gutter="20" class="slt" :style="{justifyContent:contents.searchBoxPosition=='1'?'flex-start':contents.searchBoxPosition=='2'?'center':'flex-end'}">

          <el-form-item label="零件号">
            <el-input v-model="searchForm.partNumber" placeholder="请输入零件号" clearable></el-input>
          </el-form-item>

          <el-form-item label="托号">
            <el-input v-model="searchForm.setNumber" placeholder="请输入托号" clearable></el-input>
          </el-form-item>

          <el-form-item label="MAC地址">
            <el-input v-model="searchForm.macAddress" placeholder="请输入MAC地址" clearable></el-input>
          </el-form-item>

          <el-form-item label="时间范围">
            <el-date-picker
                v-model="searchForm.timeRange"
                type="daterange"
                range-separator="至"
                start-placeholder="开始日期"
                end-placeholder="结束日期"
                clearable>
            </el-date-picker>
          </el-form-item>

          <el-form-item>
            <el-button type="primary" icon="el-icon-search" @click="search">搜索</el-button>
          </el-form-item>

          <el-form-item>
            <el-button type="primary" icon="el-icon-download" @click="exportData">批量导出数据</el-button>
          </el-form-item>


        </el-row>

      </el-form>

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
          <el-table-column prop="partNumber" label="零件号"></el-table-column>
          <el-table-column prop="setNumber" label="托号"></el-table-column>
          <el-table-column prop="macAddress" label="MAC地址"></el-table-column>
          <el-table-column prop="moduleName" label="模组名称"></el-table-column>
          <el-table-column prop="moduleSoc" label="模组SOC%"></el-table-column>
          <el-table-column prop="moduleTemperature" label="模组温度"></el-table-column>
          <el-table-column prop="wilVersion" label="WIL Version"></el-table-column>
          <el-table-column prop="scriptVersion" label="BMS Script Version"></el-table-column>
          <el-table-column prop="add_time" label="异常时间" width="180"></el-table-column>



          <el-table-column prop="exceptionStatus" label="异常状态" width="180">
            <template slot-scope="scope">
              <span v-if="scope.row.exceptionStatus === '温度异常'" style="color: #fa7d7d;">温度异常</span>
              <span v-else-if="scope.row.exceptionStatus === 'SOC异常'" style="color: red;">SOC异常</span>
              <span v-else-if="scope.row.exceptionStatus === '自放电异常'" style="color: red;">自放电异常</span>
              <span v-else-if="scope.row.exceptionStatus === '压差异常'" style="color: red;">压差异常</span>
              <span v-else style="color: red;">性能异常</span>
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

  </div>
</template>
<script>

import * as XLSX from 'xlsx';
import base from "@/utils/base";

export default {
  props: ['partNumber', 'socRange','temperature','showType','version','exceptionStatus'],
  data() {
    return {
      selectedExceptionStatus: '温度异常', // 默认选中“温度异常”
      exceptionStatuses: ['温度异常', 'SOC异常', '自放电异常','压差异常','性能异常'], // 异常状态列表
      searchForm: {
        partNumber: "",
        setNumber: "",
        macAddress: "",
        socRangeMin:0,
        socRangeMax:100,
        temperatureRangeMin:-10,
        temperatureRangeMax:40,
        showType:"",
        version:"",
        timeRange: [], // 时间范围选择器的模型
        // 其他字段保持不变
      },
      refreshInterval: null, // 用于保存定时器 ID
      refreshTime: 300000, // T 时间间隔（毫秒），定期刷新页面时间(5分钟刷新一次)
      exceptionStatus: "温度异常", // 添加异常状态筛选字段，默认温度异常
      fullDataList:[],
      dataList:[],
      paginatedDataList:[],
      pageIndex: 1,
      pageSize: 10,
      totalPage: 0,
      dataListLoading: false,
      dataListSelections: [],
      showFlag: true,
      showDetailBtn: true,
      shForm: {},
      chartVisiable: false,
      showDetail:false,
      contents:{"searchBtnFontColor":"rgba(0, 0, 0, 1)","pagePosition":"1","inputFontSize":"14px","inputBorderRadius":"0px","tableBtnDelFontColor":"#333","tableBtnIconPosition":"1","searchBtnHeight":"40px","inputIconColor":"rgba(0, 0, 0, 1)","searchBtnBorderRadius":"4px","tableStripe":true,"btnAdAllWarnFontColor":"#333","tableBtnDelBgColor":"rgba(204, 255, 102, 1)","searchBtnIcon":"1","tableSize":"medium","searchBtnBorderStyle":"groove","tableSelection":true,"searchBtnBorderWidth":"1px","tableContentFontSize":"14px","searchBtnBgColor":"#fff","inputTitleSize":"14px","btnAdAllBorderColor":"#DCDFE6","pageJumper":true,"btnAdAllIconPosition":"1","searchBoxPosition":"1","tableBtnDetailFontColor":"#333","tableBtnHeight":"40px","pagePager":true,"searchBtnBorderColor":"rgba(130, 131, 133, 1)","tableHeaderFontColor":"#909399","inputTitle":"1","tableBtnBorderRadius":"4px","btnAdAllFont":"0","btnAdAllDelFontColor":"#333","tableBtnIcon":"1","btnAdAllHeight":"40px","btnAdAllWarnBgColor":"rgba(204, 255, 102, 1)","btnAdAllBorderWidth":"1px","tableStripeFontColor":"#606266","tableBtnBorderStyle":"solid","inputHeight":"40px","btnAdAllBorderRadius":"4px","btnAdAllDelBgColor":"rgba(102, 204, 255, 1)","pagePrevNext":true,"btnAdAllAddBgColor":"rgba(117, 113, 249, 1)","searchBtnFont":"1","tableIndex":true,"btnAdAllIcon":"1","tableSortable":false,"pageSizes":true,"tableFit":true,"pageBtnBG":true,"searchBtnFontSize":"15px","tableBtnEditBgColor":"rgba(102, 204, 255, 1)","inputBorderWidth":"2px","inputFontPosition":"1","inputFontColor":"rgba(0, 0, 0, 1)","pageEachNum":10,"tableHeaderBgColor":"#fff","inputTitleColor":"#333","btnAdAllBoxPosition":"3","tableBtnDetailBgColor":"rgba(117, 113, 249, 1)","inputIcon":"1","searchBtnIconPosition":"1","btnAdAllFontSize":"14px","inputBorderStyle":"none none solid none ","inputBgColor":"rgba(117, 113, 249, 0.09)","pageStyle":false,"pageTotal":true,"btnAdAllAddFontColor":"#333","tableBtnFont":"1","tableContentFontColor":"#606266","inputBorderColor":"rgba(115, 115, 115, 1)","tableShowHeader":true,"tableBtnFontSize":"14px","tableBtnBorderColor":"#DCDFE6","inputIconPosition":"2","tableBorder":true,"btnAdAllBorderStyle":"solid","tableBtnBorderWidth":"1px","tableStripeBgColor":"#F5F7FA","tableBtnEditFontColor":"#333","tableAlign":"center"},
      layouts: '',
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
    isShowDetailBtn(newVal) {
      this.showDetailBtn = newVal;
    },
    showFlag(newVal){
      this.init();
      // this.generateRandomData();
      this.getFullDataList();
      this.dataList = this.fullDataList;
      // this.getDataList();
      //第一次加载时搜索所有数据并展示出来
      this.search();
    }
  },

  created() {
    this.init();
    // this.generateRandomData();
    this.getFullDataList();
    this.dataList = this.fullDataList;
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
  },
  methods: {
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

    init () {
      console.log("Enter ex init");
      if(this.$route.params.partNumber) {
        this.searchForm.partNumber = this.$route.params.partNumber;
        console.log("exIp:" + this.$route.params.partNumber);
      }

      if(this.$route.params.exceptionStatus) {
        this.selectedExceptionStatus = this.$route.params.exceptionStatus;
        console.log("ExceptionStatus:" + this.$route.params.exceptionStatus);
      }
    },
    // generateRandomData() {
    //   const dataList = [];
    //   const WIL_Versions = ["V1.1.4", "V2.0.0", "V2.0.4", "V2.0.8", "V2.0.9"];
    //   const BMS_Script_Versions = ["V10", "V11", "V12"];
    //   const moduleNames = {
    //     "24120503": "NCM 2P8S 模组总成 A",
    //     "24120504": "NCM 2P8S 模组总成 B",
    //     "24120149": "NCM 1P12S 模组总成A",
    //     "24120150": "NCM 1P12S 模组总成 B",
    //     "24120443": "LFP 1P12S 模组总成A",
    //     "24120444": "LFP 1P12S 模组总成 B",
    //     // 如果有更多映射，请继续添加
    //   };
    //
    //   // 获得所有moduleNames的键
    //   const partNumbers = Object.keys(moduleNames);
    //
    //   for (let i = 0; i < 500; i++) {
    //     const partNumberIndex = Math.floor(Math.random() * partNumbers.length);
    //     const partNumber = partNumbers[partNumberIndex];
    //     const setNumberSuffix = Math.floor(Math.random() * 100000).toString().padStart(5, '0');
    //     const WIL_VersionIndex = Math.floor(Math.random() * WIL_Versions.length);
    //     const BMS_Script_VersionIndex = Math.floor(Math.random() * BMS_Script_Versions.length);
    //
    //     let temperature = Math.floor(Math.random() * 31) + 10; // 默认正常温度
    //     let soc = Math.floor(Math.random() * 80) + 21; // 默认正常SOC
    //
    //     let exceptionTypes = ["温度异常", "SOC异常", "自放电异常", "通讯异常","性能异常"];
    //     let selectedException = exceptionTypes[Math.floor(Math.random() * exceptionTypes.length)];
    //
    //     let exceptionStatus = "";
    //
    //     switch (selectedException) {
    //       case "温度异常":
    //         temperature = Math.random() < 0.5 ? Math.floor(Math.random() * 11) - 10 : Math.floor(Math.random() * 6) + 35; // 生成异常温度
    //         exceptionStatus = "温度异常";
    //         break;
    //       case "SOC异常":
    //         soc = Math.floor(Math.random() * 21); // 生成异常SOC
    //         exceptionStatus = "SOC异常";
    //         break;
    //       case "自放电异常":
    //         // 温度和SOC保持正常
    //         exceptionStatus = "自放电异常";
    //         break;
    //       case "通讯异常":
    //         // 温度和SOC保持正常
    //         exceptionStatus = "通讯异常";
    //         break;
    //       case "性能异常":
    //         // 温度和SOC保持正常
    //         exceptionStatus = "性能异常";
    //         break;
    //     }
    //
    //     dataList.push({
    //       id: i + 1,
    //       partNumber: partNumber,
    //       setNumber: `BEV(1416)S${setNumberSuffix}`,
    //       macAddress: `MAC-${Math.random().toString(36).substr(2, 12)}`,
    //       moduleName: moduleNames[partNumber],
    //       moduleSoc: soc,
    //       moduleTemperature: temperature,
    //       WIL_Version: WIL_Versions[WIL_VersionIndex],
    //       BMS_Script_Version: BMS_Script_Versions[BMS_Script_VersionIndex],
    //       exceptionStatus: exceptionStatus,
    //     });
    //   }
    //   this.fullDataList = dataList;
    // },

    //修改异常数据
    updateExceptionStatus(status) {
      // 设置异常状态并触发搜索
      this.exceptionStatus = status;
      this.selectedExceptionStatus = status;
      this.search(); // 使用已经定义的search方法重新加载数据
    },
    buttonStyle(status) {
      // 根据是否被选中返回不同的样式对象
      if (this.selectedExceptionStatus === status) {
        return {
          fontSize: '18px',
          backgroundColor: '#8B0000',
          borderColor: '#8B0000',
          color: 'white',
        };
      } else {
        return {
          fontSize: '18px',
          backgroundColor: '#D3D3D3',
          borderColor: '#D3D3D3',
          color: 'black',
        };
      }
    },
    //导出数据
    exportData() {
      // 需要导出的数据
      const ws_data = this.dataList.map(item => ({
        'ID': item.id,
        '零件号': item.partNumber,
        '托号': item.setNumber,
        'MAC地址': item.macAddress,
        '模组名称': item.moduleName,
        '模组SOC%': item.moduleSoc,
        '模组温度': item.moduleTemperature,
        'WIL版本': item.wilVersion,
        '脚本版本': item.scriptVersion,
        '异常状态': item.exceptionStatus
        // 可以根据需要添加更多字段
      }));

      const ws = XLSX.utils.json_to_sheet(ws_data);
      const wb = XLSX.utils.book_new();
      XLSX.utils.book_append_sheet(wb, ws, "Data");

      // 导出文件
      XLSX.writeFile(wb, this.selectedExceptionStatus+"数据导出.xlsx");
    },

    //查询功能
    search() {
      this.dataList = this.fullDataList;
      // 使用filter方法根据搜索条件过滤fullDataList
      this.dataList = this.dataList.filter((item) => {
        let exceptionMatch = true; // 默认不过滤任何异常状态
        if (this.selectedExceptionStatus) {
          exceptionMatch = item.exceptionStatus === this.selectedExceptionStatus;
        }

        let matchesTimeRange = true;
        if (this.searchForm.timeRange && this.searchForm.timeRange.length === 2) {
          const [startTime, endTime] = this.searchForm.timeRange;
          const itemTime = new Date(item.add_time);
          matchesTimeRange = itemTime >= startTime && itemTime <= endTime;
        }

        return exceptionMatch && matchesTimeRange && (!this.searchForm.partNumber || item.partNumber === this.searchForm.partNumber) &&
            (!this.searchForm.setNumber || item.setNumber === this.searchForm.setNumber) &&
            (!this.searchForm.macAddress || item.macAddress.toUpperCase() === this.searchForm.macAddress.toUpperCase());
      });
      // 更新分页数据
      this.initPaginatedDataList();
    },

    async getFullDataList() {
      this.dataListLoading = true;

      try {
        const response = await this.$http.get("BatteryModule/exceptionInfos");
        const data = response.data;

        if (data && data.status === 0) {
          // 映射后端的异常状态码到前端的异常状态描述
          const exceptionStatusMap = {
            1: 'SOC异常',
            2: '温度异常',
            3: '自放电异常',
            4: '压差异常',
            5: '性能异常'
          };
          console.log("EXData: " + JSON.stringify(data.data,null,2));
          // 使用map函数处理每个条目，将数字状态转换为字符串描述
          this.fullDataList = data.data.map((item) => ({
            ...item,
            exceptionStatus: exceptionStatusMap[item.exceptionStatus] || '未知异常', // 使用映射，未知状态码则标为'未知异常'
          }));

          this.dataList = [...this.fullDataList];
          this.totalPage = data.total || this.fullDataList.length;
          this.search(); // 初始化数据和分页
        } else {
          this.fullDataList = [];
          this.dataList = [];
          this.totalPage = 0;
        }
      } catch (error) {
        console.error("Failed to fetch data:", error);
        this.fullDataList = [];
        this.dataList = [];
        this.totalPage = 0;
      } finally {
        this.dataListLoading = false;
      }
    },


    initPaginatedDataList(){
      this.totalPage = this.dataList.length;
      this.pageIndex = 1; // 重置到第一页
      const startIndex = (this.pageIndex - 1) * this.pageSize;
      const endIndex = startIndex + this.pageSize;
      this.paginatedDataList = this.dataList.slice(startIndex, endIndex);
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

    // // 删除
    // deleteHandler(id) {
    //   var ids = id
    //       ? [Number(id)]
    //       : this.dataListSelections.map(item => {
    //         return Number(item.id);
    //       });
    //   this.$confirm(`确定进行[${id ? "删除" : "批量删除"}]操作?`, "提示", {
    //     confirmButtonText: "确定",
    //     cancelButtonText: "取消",
    //     type: "warning"
    //   }).then(() => {
    //     this.$http({
    //       url: "BatteryModule/delete",
    //       method: "post",
    //       data: ids
    //     }).then(({ data }) => {
    //       if (data && data.code === 0) {
    //         this.$message({
    //           message: "操作成功",
    //           type: "success",
    //           duration: 1500,
    //           onClose: () => {
    //             this.search();
    //           }
    //         });
    //       } else {
    //         this.$message.error(data.msg);
    //       }
    //     });
    //   });
    // }
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


.exception-status-buttons {
  display: flex;
  justify-content: space-between;
  width: 100%; /* 确保 div 占据整行 */
  height: 60px;
}

.exception-btn {
  flex-grow: 1; /* 使按钮占据等量空间 */
  margin: 0 8px; /* 根据需要调整间距来占据整行的80% */
  font-size: 18px;
}

.exception-status-buttons .el-button--primary {
  background-color: #409EFF; /* 蓝色 */
  border-color: #409EFF;
}

.exception-status-buttons .el-button--warning {
  background-color: #640909; /* 黄色 */
  border-color: #E6A23C;
}


/* 第一个和最后一个按钮的特殊处理，以确保按钮之间的间距均匀 */
.exception-status-buttons :first-child {
  margin-left: 0;
}

.exception-status-buttons :last-child {
  margin-right: 0;
}



/* 移动端样式适配 */
@media (max-width: 768px) {
  .main-content {
    padding: 5px; /* 在小屏幕上减少内边距 */
  }

  /* 表单和按钮区域样式微调 */
  .slt, .ad {
    flex-direction: column; /* 在小屏幕上堆叠元素 */
    gap: 5px; /* 减少间距以适应更小的屏幕 */
  }

  .el-form-item {
    width: 100%; /* 让表单元素宽度自适应，避免挤压 */
  }

  .el-input__inner, .el-button {
    height: 40px; /* 对小屏幕减少元素高度 */
    font-size: 15px; /* 调整字体大小以保证可读性 */
    margin-bottom: 5px; /* 增加元素之间的间距 */
  }

  /* 调整异常状态按钮样式，提升可点击性 */
  .exception-status-buttons {
    justify-content: center; /* 中心对齐按钮 */
    flex-wrap: wrap; /* 允许按钮在必要时换行 */
    margin-bottom: 10px; /* 增加与其他元素的间距 */
  }

  .exception-btn {
    flex: 1 0 auto; /* 使按钮自动调整宽度 */
    margin: 2px; /* 减少按钮间的间距 */
    font-size: 14px; /* 调整字体大小 */
  }

  /* 表格样式调整 */
  .tables /deep/ .el-table {
    .el-table__header,
    .el-table__body {
      .el-table__row {
        th, td {
          padding: 8px; /* 减少单元格内边距 */
          font-size: 13px; /* 减小字体大小以适应更小的屏幕 */
        }
      }
    }
  }

  /* 简化分页控件样式 */
  .pages /deep/ .el-pagination {
    justify-content: center; /* 分页控件居中显示 */
    .el-pagination__prev,
    .el-pagination__next {
      font-size: 13px; /* 调整翻页按钮的字体大小 */
    }
  }
}

</style>
