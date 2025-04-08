<template>
  <div>
    <!-- 列表页 -->
    <div v-show="showFlag">
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

        </el-row>

        <el-row v-if="!isShowDetailBtn" class="ad" :style="{justifyContent:contents.btnAdAllBoxPosition=='1'?'flex-start':contents.btnAdAllBoxPosition=='2'?'center':'flex-end'}">
          <el-form-item>
            <el-button type="primary" icon="el-icon-download" @click="exportData">批量导出</el-button>
          </el-form-item>


          <el-form-item>
            <el-upload
                ref="upload"
                action="#"
                :auto-upload="false"
                accept=".xlsx, .xls"
                :on-change="handleFileChange"
                :before-upload="beforeUpload"
                :show-file-list="false"> <!-- 设置为false阻止显示文件列表 -->
              <el-button type="primary" icon="el-icon-upload">批量添加</el-button>
            </el-upload>
          </el-form-item>

          <el-form-item>
            <el-button
              :disabled="dataListSelections.length <= 0"
              type="danger"
              icon="el-icon-delete"
              @click="deleteHandler()"
            >{{ contents.btnAdAllFont == 1?'出库':'' }}</el-button>
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
          <el-table-column prop="entryStatus" label="入库状态" :formatter="formatEntryStatus"></el-table-column>
<!--          <el-table-column prop="add_time" label="记录更新时间" :formatter="formatTime"></el-table-column>-->
          <el-table-column prop="add_time" label="记录更新时间"></el-table-column>
          <el-table-column prop="module_divide" label="设备分区"></el-table-column>
          <el-table-column prop="traceCode" label="追溯码"></el-table-column>

          <el-table-column
              class="operation"
              label="操作"
              fixed="right"
              >

        <template slot-scope="scope">
          <div>
          <el-button
              type="success"
              icon="el-icon-tickets"
              size="mini"
              @click="showDetailHandler(scope.row, 'info')">
            {{ contents.tableBtnFont == 1 ? '详情' : '' }}
          </el-button>
          </div>
          <div>
       <el-button type="danger" icon="el-icon-delete" size="mini" @click="deleteHandler(scope.row.id)">{{ contents.tableBtnFont == 1?'出库':'' }}</el-button>
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
      <detail v-if="showDetail" :parent="this" :detailData="currentRowData" :isShowDetailBtn="true" ref="detail"></detail>
    </div>
</template>
<script>

import detail from "@/views/modules/BatteryModule/detail.vue";
import * as XLSX from 'xlsx';
import moment from 'moment-timezone';
import base from "@/utils/base";

export default {
  props: ['partNumber', 'socRange','temperature','showType','version','isShowDetailBtn'],
  data() {
    return {
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
        timeRange:[],// 时间范围选择器的模型
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
      //TODO 链接服务器后需要据此修改
      this.getFullDataList();
      this.dataList = this.fullDataList;
      // console.log("dataList: " + JSON.stringify(this.dataList[0], null, 2));

      // this.getDataList();
      //第一次加载时搜索所有数据并展示出来
      this.search();
      this.searchByConstraint(); // 执行搜索
    }
  },

  created() {
    this.init();
    this.getFullDataList();
  },

  mounted() {
    this.getPaginatedDataList();
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

    handleDateChange(value) {
      console.log('Selected date range:', this.searchForm.timeRange);
    },

    formatEntryStatus(row) {
      return row.entryStatus === '1' ? '库存中' : '已出库';
    },

    formatTime(time) {
      // 确保使用 moment.js 处理时区和格式化
      return moment(time).tz('Asia/Shanghai').format('YYYY-MM-DD HH:mm:ss');
    },


    // 初始化
    init () {
      if(this.$route.params.partNumber) {
        this.searchForm.partNumber = this.$route.params.partNumber;
        console.log("ip:" + this.$route.params.partNumber);
      }

      if(this.$route.params.socRange){
        console.log("sc:" + this.$route.params.socRange);
        if(this.$route.params.socRange==='0%~20%'){
          this.searchForm.socRangeMin = 0;  this.searchForm.socRangeMax = 20;
        }else if(this.$route.params.socRange==='20%~40%'){
          this.searchForm.socRangeMin = 20;  this.searchForm.socRangeMax = 40;
        }else if(this.$route.params.socRange==='40%~60%'){
          this.searchForm.socRangeMin = 40;  this.searchForm.socRangeMax = 60;
        }else if(this.$route.params.socRange==='60%~80%'){
          this.searchForm.socRangeMin = 60;  this.searchForm.socRangeMax = 80;
        }else{
          this.searchForm.socRangeMin = 80;  this.searchForm.socRangeMax = 100;
        }
      }else{
        this.searchForm.socRangeMin = 0; this.searchForm.socRangeMax = 100;
      }

      if(this.$route.params.temperature){
        console.log("tp:" + this.$route.params.temperature);
        // console.log("tp:" + this.temperature);
        if(this.$route.params.temperature==='-10°~0°'){
          this.searchForm.temperatureRangeMin = -10;  this.searchForm.temperatureRangeMax = 0;
        }else if(this.$route.params.temperature==='0°~10°'){
          this.searchForm.temperatureRangeMin = 0;  this.searchForm.temperatureRangeMax = 10;
        }else if(this.$route.params.temperature==='10°~20°'){
          this.searchForm.temperatureRangeMin = 10;  this.searchForm.temperatureRangeMax = 20;
        }else if(this.$route.params.temperature==='20°~30°'){
          this.searchForm.temperatureRangeMin = 20;  this.searchForm.temperatureRangeMax = 30;
        }else{
          this.searchForm.temperatureRangeMin = 30;  this.searchForm.temperatureRangeMax = 40;
        }
      }

      if(this.$route.params.showType){
        console.log("st:" + this.$route.params.showType);
        this.searchForm.showType = this.$route.params.showType;
      }

      if(this.$route.params.version){
        console.log("vr:" + this.$route.params.version);
        this.searchForm.version = this.$route.params.version;
      }

      // console.log("MinSOC:  " + this.searchForm.socRangeMin);
      // console.log("MaxSOC:  " + this.searchForm.socRangeMax);
      //
      // console.log("MinT:  " + this.searchForm.temperatureRangeMin);
      // console.log("MaxT:  " + this.searchForm.temperatureRangeMax);
      //
      // console.log("showType:  " + this.searchForm.showType);
      // console.log("Version:  " + this.searchForm.version);

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
    //     // 随机选择一个键
    //     const partNumberIndex = Math.floor(Math.random() * partNumbers.length);
    //     const partNumber = partNumbers[partNumberIndex];
    //     const setNumberSuffix = Math.floor(Math.random() * 100000).toString().padStart(5, '0');
    //     const WIL_VersionIndex = Math.floor(Math.random() * WIL_Versions.length);
    //     const BMS_Script_VersionIndex = Math.floor(Math.random() * BMS_Script_Versions.length);
    //
    //     dataList.push({
    //       id: i + 1,
    //       partNumber: partNumber, // 使用随机选择的键
    //       setNumber: `BEV(1416)S${setNumberSuffix}`,
    //       macAddress: `MAC-${Math.random().toString(36).substr(2, 12)}`,
    //       moduleName: moduleNames[partNumber], // 使用键对应的值
    //       moduleSoc: Math.floor(Math.random() * 100) + 1,
    //       moduleTemperature: Math.floor(Math.random() * 51) - 10,
    //       WIL_Version: WIL_Versions[WIL_VersionIndex],
    //       BMS_Script_Version: BMS_Script_Versions[BMS_Script_VersionIndex],
    //     });
    //   }
    //   this.fullDataList = dataList;
    // },




    search() {
      this.dataList = this.fullDataList;
      if (this.$route.params.isShowDetailBtn) {
        this.searchByConstraint(); // 执行更详细的搜索
      }

      this.dataList = this.dataList.filter((item) => {
        let matchesTimeRange = true;
        if (this.searchForm.timeRange && this.searchForm.timeRange.length === 2) {
          // 确保时间范围选择器的值已经是日期对象
          const startTime = moment(this.searchForm.timeRange[0]).startOf('day').toDate(); // 起始日期的午夜
          const endTime = moment(this.searchForm.timeRange[1]).endOf('day').toDate(); // 结束日期的23:59:59

          const itemTime = moment(item.add_time, "YYYY-MM-DD HH:mm:ss").tz('Asia/Shanghai').toDate();
          matchesTimeRange = itemTime >= startTime && itemTime <= endTime;
        }

        return matchesTimeRange &&
            (!this.searchForm.partNumber || item.partNumber === this.searchForm.partNumber) &&
            (!this.searchForm.setNumber || item.setNumber === this.searchForm.setNumber) &&
            (!this.searchForm.macAddress || item.macAddress.toUpperCase() === this.searchForm.macAddress.toUpperCase());
      });
      // 更新分页数据
      this.totalPage = this.dataList.length;
      this.pageIndex = 1; // 重置到第一页
      const startIndex = (this.pageIndex - 1) * this.pageSize;
      const endIndex = startIndex + this.pageSize;
      this.paginatedDataList = this.dataList.slice(startIndex, endIndex);
    },

    searchByConstraint() {
        // console.log("DATALength" + this.fullDataList.length);
        console.log("partNumber" + this.searchForm.partNumber);
        console.log("SocRange" + this.searchForm.socRangeMin + " - " + this.searchForm.socRangeMax);
        console.log("TemRange" + this.searchForm.temperatureRangeMin + " - " + this.searchForm.temperatureRangeMax);
        console.log("showType:  " + this.searchForm.showType);
        console.log("Version:  " + this.searchForm.version);
        // 使用filter方法根据搜索条件过滤fullDataList
      this.dataList = this.fullDataList.filter((item) => {
        // Check if partNumber matches
        const partNumberMatch = !this.searchForm.partNumber || item.partNumber === this.searchForm.partNumber;

        // SOC Range Condition
        const socMatch = item.moduleSoc >= this.searchForm.socRangeMin && item.moduleSoc <= this.searchForm.socRangeMax;

        // Temperature Range Condition
        const tempMatch = item.moduleTemperature >= this.searchForm.temperatureRangeMin && item.moduleTemperature <= this.searchForm.temperatureRangeMax;

        // Version or Script Version Condition based on showType
        let versionMatch = false;
        if (this.searchForm.showType === '') {
          // If showType is empty, match both version and script without specific version/BMS_Script_Version constraint
          versionMatch = true;
        } else if (this.searchForm.showType === 'version') {
          versionMatch = item.wilVersion === this.searchForm.version;
        } else if (this.searchForm.showType === 'script') {
          versionMatch = item.scriptVersion === this.searchForm.version;
        }

        return partNumberMatch && socMatch && tempMatch && versionMatch;
      });
      console.log("SearchData: " + this.dataList.length);
    },

    async getPaginatedDataList() {
      this.dataListLoading = true;
      try {
        const response = await this.$http.get(`BatteryModule/paginatedInfos`, {
          params: {
            page: this.pageIndex,
            size: this.pageSize
          }
        });
        const data = response.data;

        if (data && data.status === 0) {
          this.dataList = data.data;
          this.totalPage = data.totalItems;
        } else {
          this.$message.error('获取分页数据失败');
          this.paginatedDataList = [];
          this.totalPage = 0;
        }
      } catch (error) {
        console.error("获取数据失败:", error);
        this.$message.error("请求失败");
      } finally {
        this.dataListLoading = false;
      }
    },

    // 获取所有数据列表
    async getFullDataList() {
      // console.log("GET FULL DATA:");
      this.dataListLoading = true;

      try {
        const response = await this.$http.get("BatteryModule/infos");
        const data = response.data;

        if (data && data.status === 0) {
          this.fullDataList = data.data;
          // console.log("fullDataList: " + JSON.stringify(this.fullDataList[0], null, 2));
          // 将fullDataList的数据赋值给dataList以供操作，确保原始数据不被修改
          this.dataList = [...this.fullDataList];
          this.totalPage = data.total || data.data.length;
          // console.log("Data: " + JSON.stringify(this.dataList[0], null, 2));
          // 初始化paginatedDataList
          this.search();
        } else {
          this.fullDataList = [];
          this.dataList = [];
          this.totalPage = 0;
        }
      } catch (error) {
        console.error("Failed to fetch data:", error);
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
    // 新增、批量出库
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
      // console.log("RD: " + rowData.moduleTemperature);
      this.showDetail = true;
      this.showFlag = false;
      // 将当前行的数据存储为组件的属性
      this.currentRowData = rowData;
      console.log("show CRD" + JSON.stringify(this.currentRowData));
    },

    //批量导出数据
    exportData() {
      // 需要导出的数据列表
      const data = this.dataList.map(item => ({
        ID: item.id,
        零件号: item.partNumber,
        托号: item.setNumber,
        MAC地址: item.macAddress,
        模组名称: item.moduleName,
        模组SOC: item.moduleSoc,
        模组温度: item.moduleTemperature,
        WIL版本: item.wilVersion,
        脚本版本: item.scriptVersion,
        记录更新时间:item.add_time,
        // 可以根据需要添加更多字段
      }));

      // 使用xlsx库创建工作簿和工作表
      const ws = XLSX.utils.json_to_sheet(data);
      const wb = XLSX.utils.book_new();
      XLSX.utils.book_append_sheet(wb, ws, "Data");

      // 生成Excel文件并保存
      XLSX.writeFile(wb, "DataListExport.xlsx");
    },

    //模组批量上传
    beforeUpload(file) {
      const isExcel = file.type === 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' || file.type === 'application/vnd.ms-excel';
      if (!isExcel) {
        this.$message.error('文件格式错误，请上传Excel文件！');
        return false;
      }
      return true;
    },

    handleFileChange(file) {
      const reader = new FileReader();
      reader.onload = e => {
        try {
          const data = e.target.result;
          const workbook = XLSX.read(data, {type: 'binary'});
          const sheetName = workbook.SheetNames[0];
          const worksheet = workbook.Sheets[sheetName];
          const json = XLSX.utils.sheet_to_json(worksheet, {header: 1});
          this.validateAndProcessExcelData(json);
        } catch (err) {
          this.$message.error('文件解析失败！');
        }
      };
      reader.readAsBinaryString(file.raw);
    },

    // 验证并处理Excel数据
    validateAndProcessExcelData(data) {
      const headers = data[0]; // 第一行应该包含表头
      const rows = data.slice(1); // 其余行是数据

      const batteryModulesDTO = rows.map(row => {
        // 通过表头和行数据创建对象
        const obj = headers.reduce((acc, header, index) => {
          acc[header] = row[index] || ''; // 使用空字符串作为缺失值的默认值
          return acc;
        }, {});

        return {
          partNumber: obj['模组BC'],
          setNumber: obj['套号'],
          moduleName: obj['总成'],
          macAddress: obj['MAC'],
          moduleSoc: '',
          moduleTemperature: '',
          wilVersion: '',
          scriptVersion: '',
          module_divide: '',
          // add_time: moment().format('YYYY-MM-DD HH:mm:ss'),
          add_time: '',
          traceCode:obj['trace_code'],
        };
      });

      if (batteryModulesDTO.length > 0) {
        this.batchAddModules(batteryModulesDTO);
      } else {
        this.$message.error('无有效数据可以添加');
      }
    },


    batchAddModules(batteryModulesDTO) {
      // 向后端发送封装好的 BatteryModuleDTO 列表
      this.$http.post('/BatteryModule/batchAdd', batteryModulesDTO)
          .then(response => {
            // 根据后端响应结构来处理消息
            if (response.data.status === 0) { // 假设 status: 0 表示成功
              this.$message.success('批量添加成功！');
              // 刷新前端数据列表
              this.refreshDataList();
            } else {
              this.$message.error('批量添加失败：' + response.data.message);
            }
          })
          .catch(error => {
            this.$message.error('请求失败：' + error.message);
          });
    },
    // 添加一个新的方法来刷新数据列表
    refreshDataList() {
      this.getFullDataList().then(() => {
        // 可能还需要处理其他状态更新逻辑，如重置分页等
        this.pageIndex = 1;
        this.pageSize = 10; // 或者保持当前的 pageSize
        this.initPaginatedDataList();
      });
    },

    // 出库
    deleteHandler(id) {
      // 如果提供了id参数，使用该id；否则，从选中项构造ids数组
      const ids = id ? [id] : this.dataListSelections.map(item => item.id);
      console.log("IDS:  " + ids);

      this.$confirm(`确定进行[${id ? "出库" : "批量出库"}]操作?`, "提示", {
        confirmButtonText: "确定",
        cancelButtonText: "取消",
        type: "warning"
      }).then(() => {
        this.$http({
          url: "BatteryModule/delete",
          method: "post",
          data: JSON.stringify(ids),  // 将ids数组转换为JSON字符串
          headers: {
            'Content-Type': 'application/json'  // 设置请求头
          }
        }).then(({ data }) => {
          if (data && data.status === 0) {
            this.$message({
              message: "出库成功",  // 修改提示消息
              type: "success",
              duration: 1500,
              onClose: () => {
                this.getFullDataList();
              }
            });
          } else {
            this.$message.error(data.msg || "操作失败");  // 使用后端返回的错误消息或默认消息
          }
        }).catch(error => {
          console.error("请求失败: ", error);
          this.$message.error("请求异常");
        });
      }).catch(() => {
        // 用户取消操作，不需要额外处理
      });
    },

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

  & /deep/  .operation {
    width: 260px; /* PC端固定宽度为260px */
  }
  }


  @media (max-width: 768px) {
    /* 搜索区域样式调整 */
    .form-content .slt, .form-content .ad {
      flex-direction: column;
      align-items: stretch;
    }

    .form-content .el-form-item {
      width: 100%; /* 占满整行 */
      margin-bottom: 10px; /* 为堆叠的表单项添加底部边距 */
      display: flex;
      flex-direction: column;
    }

    .form-content .el-form-item .el-form-item__label {
      width: auto; /* 自适应宽度 */
      text-align: left; /* 左对齐文本 */
      margin-bottom: 4px; /* 与输入框间隔 */
    }

    .form-content .el-input__inner,
    .form-content .el-select__inner,
    .form-content .el-date-editor .el-input__inner {
      width: 100%; /* 输入框宽度占满可用空间 */
    }

    .form-content .el-button {
      width: 100%; /* 使按钮宽度占满可用空间 */
      padding: 8px; /* 适当增减内边距 */
      font-size: 12px; /* 减小字体大小 */
    }

    .form-content .el-upload {
      width: 100%; /* 使按钮宽度占满可用空间 */
      padding: 8px; /* 适当增减内边距 */
      font-size: 12px; /* 减小字体大小 */
    }

    /* 表格和分页样式调整，请保持之前的调整 */
  }

</style>
