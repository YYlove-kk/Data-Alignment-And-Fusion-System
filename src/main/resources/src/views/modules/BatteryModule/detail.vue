<style lang="scss" scoped>
.editor {
  height: auto; // 自适应高度
}

.addEdit-block {
  padding: 24px;
  background-color: #ffffff;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
  border-radius: 8px;
  margin: 20px;
}

.detail-form-content {
  padding: 20px;
  background-color: #f7f7f7;
  border-radius: 8px;
}

.chart-container {
  width: 100%;
  height: 300px;
  margin-top: 20px;
}

.chart-form-item {
  margin-bottom: 20px;
}

.chart-volt{
  width: 100%;
  height:400px;
}

.btn-close-back {
  background-color: #f56c6c;
  border-color: #f56c6c;
  &:hover {
    background-color: darken(#f56c6c, 5%);
    border-color: darken(#f56c6c, 5%);
  }
  /* 使按钮居中 */
  display: flex; /* 或者使用 flex */
  margin: 20px auto; /* 上下保持20px间距，自动调整左右间距以居中 */
  padding: 20px 40px; /* 根据需要调整内间距 */
  font-size: 20px;
  text-align: center;
  color: #fff;
  cursor: pointer;
  border-radius: 4px; /* 根据需要调整边角圆滑度 */
}

.btn-close-back.centered-button {
  background-color: #f56c6c;
  border-color: #f56c6c;
  &:hover {
    background-color: darken(#f56c6c, 5%);
    border-color: darken(#f56c6c, 5%);
  }
  /* 使按钮居中 */
  display: flex; /* 或者使用 flex */
  margin: 10px auto; /* 上下保持20px间距，自动调整左右间距以居中 */
  padding: 10px 20px; /* 根据需要调整内间距 */
  font-size: 20px;
  text-align: center;
  color: #fff;
  cursor: pointer;
  border-radius: 4px; /* 根据需要调整边角圆滑度 */
}

.single-col-form-item, .chart-form-item {
  padding: 15px;
  background-color: #08afea;
  border-radius: 8px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.05);
}

.input, .single-input {
  border-radius: 4px;
}

.btn {
  margin-top: 20px;
  display: flex; /* 设置为Flex布局 */
  justify-content: center; /* 水平居中 */
}

/* 移动端样式适配 */
@media (max-width: 768px) {
  .addEdit-block {
    margin: 10px; /* 缩小外边距 */
    padding: 15px; /* 减少内边距 */
  }

  .detail-form-content {
    padding: 10px; /* 减少内边距 */
  }

  .chart-container {
    height: 200px; /* 减小图表高度 */
    overflow-x: scroll; /* 允许横向滚动，展示完整的图表 */
  }

  .echarts-bar .bar {
    barWidth: '40%'; /* 移动端适应，柱状图稍宽 */
  }

  .btn-close-back {
    padding: 10px 20px; /* 调整按钮内边距 */
    font-size: 16px; /* 减小字体大小 */
  }

  .el-form-item {
    /* 优化移动端的表单项布局 */
    flex-direction: row; /* 表单标签和输入框水平排列 */
    justify-content: space-between; /* 分散对齐，确保标签和输入框间距 */
    align-items: center; /* 垂直居中，提高美观度 */
  }

  .el-row {
    margin-bottom: 0; /* 移除行间距 */
  }

  .el-col {
    margin-bottom: 15px; /* 增加列的底部间距 */
  }

  .el-input__inner, .el-select__inner, .el-date-editor .el-input__inner {
    border-radius: 4px; /* 统一输入框和选择框的边角圆滑度 */
  }

  /* 调整标签的布局，使其在输入框上方显示 */
  .el-form-item__label {
    float: none;
    display: block;
    text-align: left; /* 左对齐标签文本 */
    margin-bottom: 5px; /* 增加标签与输入框的间距 */
  }

  /* 调整输入框和选择框的宽度 */
  .el-input, .el-select {
    width: 100%; /* 输入框和选择框宽度占满可用空间 */
  }

  /* 图表容器样式调整 */
  [ref="volChart"], [ref="tempChart"], [ref="socChart"] {
    width: 100%; /* 图表宽度占满可用空间 */
    height: 200px; /* 适当调整图表高度 */
  }

  /* 按钮居中显示 */
  .btn {
    justify-content: center; /* Flex布局下按钮水平居中 */
  }
}

</style>

<template>
  <div class="addEdit-block">
    <el-form
        ref="ruleForm"
        :model="ruleForm"
        :rules="rules"
        label-width="100px"
        class="detail-form-content"
    >
      <el-row :gutter="20">
        <el-form-item label="模组零件号">
          <el-input
              v-model="ruleForm.partNumber"
              placeholder="模组编号"
              readonly
          />
        </el-form-item>
        <el-form-item label="模组名称">
          <el-input
              v-model="ruleForm.moduleName"
              placeholder="模组名称"
              readonly
          />
        </el-form-item>

        <el-form-item label="MAC地址">
          <el-input
              v-model="ruleForm.macAddress"
              placeholder="MAC地址"
              readonly
          />
        </el-form-item>

        <el-form-item label="模组SOC">
          <el-input
              v-model="ruleForm.moduleSoc"
              placeholder="SOC值"
              readonly
          />
        </el-form-item>

        <el-form-item label="模组温度">
          <el-input
              v-model="ruleForm.moduleTemperature"
              placeholder="模组温度"
              readonly
          />
        </el-form-item>

        <!--显示最大电芯电压差值 -->
        <el-form-item label="电芯电压最大差值(mv)">
          <el-input
              v-model="maxVoltageDifference"
              placeholder="电芯电压最大差值"
              readonly
          />
        </el-form-item>

        <el-form-item label="模组电压" class="single-col-form-item">
          <el-input
              v-model="ruleForm.moduleVolt"
              placeholder="模组电压"
              readonly
          />
        </el-form-item>

        <div>
        <el-col :span="24">
          <el-form-item label="电芯电压">
            <!-- ECharts图表容器 -->
            <div ref="volChart" class="chart-container"></div>
          </el-form-item>
        </el-col>
        </div>

        <div>
        <el-col :span="24">
          <el-form-item label="历史温度数据变化曲线">
            <div ref="tempChart" class="chart-container"></div>
          </el-form-item>
        </el-col>
        </div>
        <div>
        <el-col :span="24">
          <el-form-item label="历史SOC数据变化曲线">
            <div ref="socChart" class="chart-container"></div>
          </el-form-item>
        </el-col>
        </div>

        <el-button class="centered-button" @click="exportToExcel">导出历史数据</el-button>

      </el-row>

      <el-row>
        <el-col :span="24">
          <el-form-item label="FPA Version">
            <el-input v-model="FPA" placeholder="FPA版本" readonly />
          </el-form-item>
        </el-col>

        <el-col :span="24">
          <el-form-item label="WIL Version">
            <el-input v-model="ruleForm.wilVersion" placeholder="请输入WIL版本" readonly />
          </el-form-item>
        </el-col>

        <el-col :span="24">
          <el-form-item label="BMS Script Version">
            <el-input v-model="ruleForm.scriptVersion" placeholder="请输入BMS脚本版本" readonly />
          </el-form-item>
        </el-col>

        <el-col :span="24">
          <el-form-item label="状态更新时间">
            <el-input v-model="ruleForm.add_time" placeholder="状态更新时间" readonly />
          </el-form-item>
        </el-col>

      </el-row>


      <el-button class="btn-close-back" @click="back()">返回</el-button>


    </el-form>
  </div>
</template>

<script>
// 数字，邮件，手机，url，身份证校验
import { isNumber,isIntNumer,isEmail,isPhone, isMobile,isURL,checkIdCard } from "@/utils/validate";
import * as echarts from 'echarts';
import moment from 'moment';
import * as XLSX from 'xlsx';


export default {
  props: ["parent",'detailData','isShowDetailBtn'],
  data() {
    return {
    // socData: [3.5,2,1,3.1,3.2,3.5,4.0,3.1,3.2,3.5],
      socData: [3.5,2,1,3.1,3.2,3.5,4.0,3.1,3.2,3.5,3.4,3.3,3.0,2.8],
      FPA: "V1.X.X",
      version: "V1.0.0",
      scriptVersion: "V10",
      config:"XXX",
      chartInstance: null,
      maxVoltageDifference: '', // 用于存储电芯电压的最大差值
      configDialogVisible: false, // 配置文件对话框可见性
      configContent: '此处显示配置文件',
      addEditForm: {"btnSaveFontColor":"#fff","selectFontSize":"14px","btnCancelBorderColor":"#DCDFE6","inputBorderRadius":"4px","inputFontSize":"14px","textareaBgColor":"#fff","btnSaveFontSize":"14px","textareaBorderRadius":"4px","uploadBgColor":"#fff","textareaBorderStyle":"solid","btnCancelWidth":"88px","textareaHeight":"120px","dateBgColor":"#fff","btnSaveBorderRadius":"4px","uploadLableFontSize":"14px","textareaBorderWidth":"1px","inputLableColor":"#606266","addEditBoxColor":"#fff","dateIconFontSize":"14px","btnSaveBgColor":"rgba(117, 113, 249, 1)","uploadIconFontColor":"#8c939d","textareaBorderColor":"#DCDFE6","btnCancelBgColor":"#ecf5ff","selectLableColor":"#606266","btnSaveBorderStyle":"solid","dateBorderWidth":"1px","dateLableFontSize":"14px","dateBorderRadius":"4px","btnCancelBorderStyle":"solid","selectLableFontSize":"14px","selectBorderStyle":"solid","selectIconFontColor":"#C0C4CC","btnCancelHeight":"44px","inputHeight":"40px","btnCancelFontColor":"#606266","dateBorderColor":"#DCDFE6","dateIconFontColor":"#C0C4CC","uploadBorderStyle":"solid","dateBorderStyle":"solid","dateLableColor":"#606266","dateFontSize":"14px","inputBorderWidth":"1px","uploadIconFontSize":"28px","selectHeight":"40px","inputFontColor":"#606266","uploadHeight":"148px","textareaLableColor":"#606266","textareaLableFontSize":"14px","btnCancelFontSize":"14px","inputBorderStyle":"solid","btnCancelBorderRadius":"4px","inputBgColor":"#fff","inputLableFontSize":"14px","uploadLableColor":"#606266","uploadBorderRadius":"4px","btnSaveHeight":"44px","selectBgColor":"#fff","btnSaveWidth":"88px","selectIconFontSize":"14px","dateHeight":"40px","selectBorderColor":"#DCDFE6","inputBorderColor":"#DCDFE6","uploadBorderColor":"#DCDFE6","textareaFontColor":"#606266","selectBorderWidth":"1px","dateFontColor":"#606266","btnCancelBorderWidth":"1px","uploadBorderWidth":"1px","textareaFontSize":"14px","selectBorderRadius":"4px","selectFontColor":"#606266","btnSaveBorderColor":"rgba(117, 113, 249, 1)","btnSaveBorderWidth":"1px"},
      id: '',
      type: '',
      ruleForm: {
        partNumber: this.getUUID(),
        moduleName: '',
        macAddress:'',
        moduleSoc:'',
        moduleTemperature:'',
        moduleVolt:'',
        add_time:'',
      },
      rules: {
          partNumber: [],
          mduleName: [],
      },
      minSoc:20,//电芯电压阈值最低点
      maxSoc:60,//电芯电压阈值最高点
      voltageDiffThreshold:30,//电芯电压差的阈值
      historyData: [],
    };
  },

  watch: {
    detailData: {
      immediate: true,
      handler(newVal) {
        if (newVal) {
          this.ruleForm = { // 直接使用传递的数据初始化ruleForm
            partNumber: newVal.partNumber,
            moduleName: newVal.moduleName,
            macAddress: newVal.macAddress,
            moduleSoc: newVal.moduleSoc,
            moduleTemperature: newVal.moduleTemperature,
            wilVersion: newVal.wilVersion,
            scriptVersion: newVal.scriptVersion,
            moduleVolt:newVal.moduleVolt,
            add_time: newVal.add_time,
          };
          this.fetchBatteryCellInfo();
          this.fetchHistoryData();
        }
      }
    }
  },

  mounted() {
    // 在组件挂载后立即获取历史数据
    // this.getSocData();
    this.initChart();
    // this.fetchBatteryCellInfo(); // 在组件挂载时获取电芯信息
    // this.fetchFPA(); // 在组件创建时获取 FPA 数据
    this.fetchVersion(); // 在组件创建时获取 Version 数据
    this.fetchConfig(); // 在组件创建时获取 Config 数据
  },
  computed: {},
  created() {
    this.addEditStyleChange()
    this.addEditUploadStyleChange()
  },
  methods: {
    formatTime(time) {
      return moment(time).tz('Asia/Shanghai').format('YYYY-MM-DD HH:mm:ss');
    },

    //导出历史数据
    exportToExcel() {
      const ws_data = [
        ['模组零件号', '模组名称', 'MAC地址', '模组SOC', '模组温度', '数据记录时间'],
        ...this.historyData.map(item => [
          this.ruleForm.partNumber,
          this.ruleForm.moduleName,
          this.ruleForm.macAddress,
          item.moduleSoc,
          item.moduleTemperature,
          this.formatTime(item.add_time) // 确保使用formatTime来格式化时间
        ])
      ];

      const ws = XLSX.utils.aoa_to_sheet(ws_data);
      const wb = XLSX.utils.book_new();
      XLSX.utils.book_append_sheet(wb, ws, "HistoryData");

      // 生成Excel文件并下载
      XLSX.writeFile(wb, `HistoryData-${this.ruleForm.partNumber}.xlsx`);
    },

        // 下载
    download(file){
      window.open(`${file}`)
    },
    // 初始化
    init(id) {
      if (id) {
        this.id = id;
        this.info(id);
      }
      // // 获取模组信息
      // this.$http({
      //   url: `${this.$storage.get('sessionTable')}/session`,
      //   method: "get"
      // }).then(({ data }) => {
      //   if (data && data.code === 0) {
      //     var json = data.data;
      //   }else {
      //     this.$message.error(data.msg);
      //   }
      // });
    },
    // 多级联动参数
    info(id) {
      this.$http({
        url: `BatteryModuleInfo/info/${id}`,
        method: "get"
      }).then(({ data }) => {
        if (data && data.code === 0) {
          this.ruleForm = data.data;
          console.log("ruleForm:'\n'"+JSON.stringify(this.ruleForm, null, 2));
        } else {
          this.$message.error(data.msg);
        }
      });
    },
        // 提交
    // 获取uuid
    getUUID () {
      return new Date().getTime();
    },
    // 返回
    back() {
      this.parent.showFlag = true;
      this.parent.showDetail = false;
      this.parent.contentStyleChange();
      this.$emit('closeDetail'); // 通知父组件关闭详情
    },
    addEditStyleChange() {
	  this.$nextTick(()=>{
	    // input
	    document.querySelectorAll('.addEdit-block .input .el-input__inner').forEach(el=>{
	      el.style.height = this.addEditForm.inputHeight
	      el.style.color = this.addEditForm.inputFontColor
	      el.style.fontSize = this.addEditForm.inputFontSize
	      el.style.borderWidth = this.addEditForm.inputBorderWidth
	      el.style.borderStyle = this.addEditForm.inputBorderStyle
	      el.style.borderColor = this.addEditForm.inputBorderColor
	      el.style.borderRadius = this.addEditForm.inputBorderRadius
	      el.style.backgroundColor = this.addEditForm.inputBgColor
	    })
	    document.querySelectorAll('.addEdit-block .input .el-form-item__label').forEach(el=>{
	      el.style.lineHeight = this.addEditForm.inputHeight
	      el.style.color = this.addEditForm.inputLableColor
	      el.style.fontSize = this.addEditForm.inputLableFontSize
	    })
	    // select
	    document.querySelectorAll('.addEdit-block .select .el-input__inner').forEach(el=>{
	      el.style.height = this.addEditForm.selectHeight
	      el.style.color = this.addEditForm.selectFontColor
	      el.style.fontSize = this.addEditForm.selectFontSize
	      el.style.borderWidth = this.addEditForm.selectBorderWidth
	      el.style.borderStyle = this.addEditForm.selectBorderStyle
	      el.style.borderColor = this.addEditForm.selectBorderColor
	      el.style.borderRadius = this.addEditForm.selectBorderRadius
	      el.style.backgroundColor = this.addEditForm.selectBgColor
	    })
	    document.querySelectorAll('.addEdit-block .select .el-form-item__label').forEach(el=>{
	      el.style.lineHeight = this.addEditForm.selectHeight
	      el.style.color = this.addEditForm.selectLableColor
	      el.style.fontSize = this.addEditForm.selectLableFontSize
	    })
	    document.querySelectorAll('.addEdit-block .select .el-select__caret').forEach(el=>{
	      el.style.color = this.addEditForm.selectIconFontColor
	      el.style.fontSize = this.addEditForm.selectIconFontSize
	    })
	    // date
	    document.querySelectorAll('.addEdit-block .date .el-input__inner').forEach(el=>{
	      el.style.height = this.addEditForm.dateHeight
	      el.style.color = this.addEditForm.dateFontColor
	      el.style.fontSize = this.addEditForm.dateFontSize
	      el.style.borderWidth = this.addEditForm.dateBorderWidth
	      el.style.borderStyle = this.addEditForm.dateBorderStyle
	      el.style.borderColor = this.addEditForm.dateBorderColor
	      el.style.borderRadius = this.addEditForm.dateBorderRadius
	      el.style.backgroundColor = this.addEditForm.dateBgColor
	    })
	    document.querySelectorAll('.addEdit-block .date .el-form-item__label').forEach(el=>{
	      el.style.lineHeight = this.addEditForm.dateHeight
	      el.style.color = this.addEditForm.dateLableColor
	      el.style.fontSize = this.addEditForm.dateLableFontSize
	    })
	    document.querySelectorAll('.addEdit-block .date .el-input__icon').forEach(el=>{
	      el.style.color = this.addEditForm.dateIconFontColor
	      el.style.fontSize = this.addEditForm.dateIconFontSize
	      el.style.lineHeight = this.addEditForm.dateHeight
	    })
	    // upload
	    let iconLineHeight = parseInt(this.addEditForm.uploadHeight) - parseInt(this.addEditForm.uploadBorderWidth) * 2 + 'px'
	    document.querySelectorAll('.addEdit-block .upload .el-upload--picture-card').forEach(el=>{
	      el.style.width = this.addEditForm.uploadHeight
	      el.style.height = this.addEditForm.uploadHeight
	      el.style.borderWidth = this.addEditForm.uploadBorderWidth
	      el.style.borderStyle = this.addEditForm.uploadBorderStyle
	      el.style.borderColor = this.addEditForm.uploadBorderColor
	      el.style.borderRadius = this.addEditForm.uploadBorderRadius
	      el.style.backgroundColor = this.addEditForm.uploadBgColor
	    })
	    document.querySelectorAll('.addEdit-block .upload .el-form-item__label').forEach(el=>{
	      el.style.lineHeight = this.addEditForm.uploadHeight
	      el.style.color = this.addEditForm.uploadLableColor
	      el.style.fontSize = this.addEditForm.uploadLableFontSize
	    })
	    document.querySelectorAll('.addEdit-block .upload .el-icon-plus').forEach(el=>{
	      el.style.color = this.addEditForm.uploadIconFontColor
	      el.style.fontSize = this.addEditForm.uploadIconFontSize
	      el.style.lineHeight = iconLineHeight
	      el.style.display = 'block'
	    })
	    // 多文本输入框
	    document.querySelectorAll('.addEdit-block .textarea .el-textarea__inner').forEach(el=>{
	      el.style.height = this.addEditForm.textareaHeight
	      el.style.color = this.addEditForm.textareaFontColor
	      el.style.fontSize = this.addEditForm.textareaFontSize
	      el.style.borderWidth = this.addEditForm.textareaBorderWidth
	      el.style.borderStyle = this.addEditForm.textareaBorderStyle
	      el.style.borderColor = this.addEditForm.textareaBorderColor
	      el.style.borderRadius = this.addEditForm.textareaBorderRadius
	      el.style.backgroundColor = this.addEditForm.textareaBgColor
	    })
	    document.querySelectorAll('.addEdit-block .textarea .el-form-item__label').forEach(el=>{
	      // el.style.lineHeight = this.addEditForm.textareaHeight
	      el.style.color = this.addEditForm.textareaLableColor
	      el.style.fontSize = this.addEditForm.textareaLableFontSize
	    })
	    // 保存
	    document.querySelectorAll('.addEdit-block .btn .btn-success').forEach(el=>{
	      el.style.width = this.addEditForm.btnSaveWidth
	      el.style.height = this.addEditForm.btnSaveHeight
	      el.style.color = this.addEditForm.btnSaveFontColor
	      el.style.fontSize = this.addEditForm.btnSaveFontSize
	      el.style.borderWidth = this.addEditForm.btnSaveBorderWidth
	      el.style.borderStyle = this.addEditForm.btnSaveBorderStyle
	      el.style.borderColor = this.addEditForm.btnSaveBorderColor
	      el.style.borderRadius = this.addEditForm.btnSaveBorderRadius
	      el.style.backgroundColor = this.addEditForm.btnSaveBgColor
	    })
	    // 返回
	    document.querySelectorAll('.addEdit-block .btn .btn-close').forEach(el=>{
	      el.style.width = this.addEditForm.btnCancelWidth
	      el.style.height = this.addEditForm.btnCancelHeight
	      el.style.color = this.addEditForm.btnCancelFontColor
	      el.style.fontSize = this.addEditForm.btnCancelFontSize
	      el.style.borderWidth = this.addEditForm.btnCancelBorderWidth
	      el.style.borderStyle = this.addEditForm.btnCancelBorderStyle
	      el.style.borderColor = this.addEditForm.btnCancelBorderColor
	      el.style.borderRadius = this.addEditForm.btnCancelBorderRadius
	      el.style.backgroundColor = this.addEditForm.btnCancelBgColor
	    })
	  })
	},
	  addEditUploadStyleChange() {
		this.$nextTick(()=>{
		  document.querySelectorAll('.addEdit-block .upload .el-upload-list--picture-card .el-upload-list__item').forEach(el=>{
			el.style.width = this.addEditForm.uploadHeight
			el.style.height = this.addEditForm.uploadHeight
			el.style.borderWidth = this.addEditForm.uploadBorderWidth
			el.style.borderStyle = this.addEditForm.uploadBorderStyle
			el.style.borderColor = this.addEditForm.uploadBorderColor
			el.style.borderRadius = this.addEditForm.uploadBorderRadius
			el.style.backgroundColor = this.addEditForm.uploadBgColor
		  })
	  })
	},

    // 获取电芯信息
    fetchBatteryCellInfo() {
      const macAddress = this.ruleForm.macAddress; // Assume MAC address is available
      this.$http.get(`/BatteryCell/getInfosByMac`, {
        params: { macAddress: macAddress }
      }).then(response => {
        const { data } = response;
        if (data && data.status === 0) {
          // Extract battery voltage data
          const batteryVoltData = data.data.map(item => item.batteryVolt);
          // Update the chart with new data
          this.initChart(batteryVoltData);

          // 计算最大和最小电压差值
          const maxVoltage = Math.max(...batteryVoltData);
          const minVoltage = Math.min(...batteryVoltData);
          this.maxVoltageDifference = (maxVoltage - minVoltage).toFixed(2); // 保留两位小数
        } else {
          this.$message.error(data.message || '获取模组信息失败');
        }
      }).catch(error => {
        console.error('Error fetching battery cell information:', error);
        this.$message.error('Failed to fetch battery cell information');
      });
    },


    // 添加获取历史数据的方法
    fetchHistoryData() {
      // 假设有一个方法向后端请求数据，这里需要根据实际情况调整
      const endTime = moment().format('YYYY-MM-DD HH:mm:ss');
      const startTime = moment().subtract(6, 'weeks').startOf('isoWeek').format('YYYY-MM-DD HH:mm:ss');

      console.log("His-st:" + startTime);
      console.log("His-et:" + endTime);
      console.log("His-mac:" + this.ruleForm.macAddress);

      const historyBatteryModuleDTO = JSON.stringify({
        macAddress: this.ruleForm.macAddress,
        startTime: startTime,
        endTime: endTime
      });

      // console.log("His-st:" + startTime);
      // console.log("His-et:" + endTime);
      // console.log("His-mac:" + this.ruleForm.macAddress);

      // 直接将数据作为请求体发送
      this.$http.post('/HistoryBatteryModule/infos', historyBatteryModuleDTO,{
        headers: {
          'Content-Type': 'application/json'
        },
      }).then(response => {
        const data = response.data;
        if(data && data.status === 0) {
          console.log("Data---:", JSON.stringify(data.data, null, 2));
          this.historyData = data.data;
          // 将后端返回的数据格式化为图表所需格式
          const dates = this.historyData.map(item => item.add_time);
          const temperatures = this.historyData.map(item => item.moduleTemperature);
          const socs = this.historyData.map(item => item.moduleSoc);
          this.initTempChart(temperatures, dates); // 初始化温度图表
          this.initSocChart(socs, dates); // 初始化SOC图表
        } else {
          console.error('获取历史数据失败');
        }
      }).catch(error => {
        console.error('Error fetching history data:', error);
      });

    },

    // 更新电压图表的方法（示例）
    updateVoltageChart(batteryVoltData) {
      // 假设你有一个 ECharts 图表实例
      // 这里你需要根据实际的图表配置进行调整
      const myChart = this.chartInstance; // 假设这是你的图表实例
      const option = {
        // 图表配置
        series: [{
          // 系列配置
          data: batteryVoltData // 电压数据
        }]
      };
      myChart.setOption(option);
    },

    // 初始化温度图表
    initTempChart(temperatureData, dates) {
      const tempChart = echarts.init(this.$refs.tempChart);
      const tempOption = {
        tooltip: {
          trigger: 'axis'
        },
        xAxis: {
          type: 'category',
          data: dates
        },
        yAxis: {
          type: 'value',
          name: '温度 (°C)'
        },
        series: [{
          data: temperatureData,
          type: 'line',
          name: '温度'
        }]
      };
      tempChart.setOption(tempOption);
    },

// 初始化SOC图表
    initSocChart(socData, dates) {
      const socChart = echarts.init(this.$refs.socChart);
      const socOption = {
        tooltip: {
          trigger: 'axis'
        },
        xAxis: {
          type: 'category',
          data: dates
        },
        yAxis: {
          type: 'value',
          name: 'SOC (%)'
        },
        series: [{
          data: socData,
          type: 'line',
          name: 'SOC'
        }]
      };
      socChart.setOption(socOption);
    },



    initChart(batteryVoltData) {
      const myChart = echarts.init(this.$refs.volChart);

      // 标记需要变色的电芯
      // const colorFlags = batteryVoltData.map((voltage, index, array) => {
      //   // 检查是否超过阈值
      //   if (voltage < this.minSoc || voltage > this.maxSoc) {
      //     return 'red'; // 超过阈值设为红色
      //   }
      //
      //   // 检查与其他电芯电压是否有超过30mv的差值
      //   for (let i = 0; i < array.length; i++) {
      //     if (array[i] >= this.minSoc && array[i] <= this.maxSoc && i !== index && Math.abs(voltage - array[i]) >= this.voltageDiffThreshold) {
      //       return 'yellow'; // 与其他电芯差值超过30mv设为黄色
      //     }
      //   }
      //
      //   return 'green'; // 默认为绿色
      // });
      const maxBatteryVoltage = Math.max(...batteryVoltData);
      const minBatteryVoltage = Math.min(...batteryVoltData);
      const maxVoltageIndex = batteryVoltData.indexOf(maxBatteryVoltage);
      const minVoltageIndex = batteryVoltData.indexOf(minBatteryVoltage);

      const colorFlags = batteryVoltData.map((voltage, index) => {
        // 检查是否超过阈值
        if (voltage < this.minSoc || voltage > this.maxSoc) {
          return 'red'; // 超过阈值设为红色
        }

        // 如果电压差超过设定的阈值，只标记最大和最小电压为红色
        if (maxBatteryVoltage - minBatteryVoltage > this.voltageDiffThreshold) {
          if (index === maxVoltageIndex || index === minVoltageIndex) {
            return 'red';
          }
        }

        return 'green'; // 默认为绿色
      });

      const option = {
        tooltip: {},
        xAxis: {
          type: 'category',
          data: batteryVoltData.map((_, index) => `Cell ${index + 1}`),
        },
        yAxis: {
          type: 'value',
          min: 0,
          max: 'dataMax',
        },
        series: [
          {
            data: batteryVoltData.map((voltage, index) => ({
              value: voltage,
              itemStyle: { color: colorFlags[index] }
            })),
            type: 'bar',
            showBackground: true,
            backgroundStyle: {
              color: 'rgba(180, 180, 180, 0.2)',
            },
          },
        ],
      };

      myChart.setOption(option);
    },


    fetchFPA() {
      this.$http.get(`BatteryModuleInfo/FPA/${this.id}`).then(response => {
        const data = response.data;
        if (data && data.code === 0) {
          this.FPA = data.data; // 将获取的 FPA 数据赋值给 FPA
        } else {
          this.$message.error(data.msg); // 如果获取失败，显示错误信息
        }
      }).catch(error => {
        console.error('Error fetching FPA data:', error);
      });
    },

    fetchVersion() {
      this.$http.get(`BatteryModuleInfo/version/${this.id}`).then(response => {
        const data = response.data;
        if (data && data.code === 0) {
          this.version = data.data; // 将获取的 version 数据赋值给 version
        } else {
          this.$message.error(data.msg); // 如果获取失败，显示错误信息
        }
      }).catch(error => {
        console.error('Error fetching Version data:', error);
      });
    },

    fetchConfig() {
      this.$http.get(`BatteryModuleInfo/config/${this.id}`).then(response => {
        const data = response.data;
        if (data && data.code === 0) {
          this.config = data.data; // 将获取的 version 数据赋值给 version
        } else {
          this.$message.error(data.msg); // 如果获取失败，显示错误信息
        }
      }).catch(error => {
        console.error('Error fetching Config data:', error);
      });
    },

    openConfigDialog() {
      //todo 向后端发送HTTP请求获取
      this.configContent = '此处显示配置文件'; // 重置配置文件内容
      this.configDialogVisible = true; // 打开配置文件对话框
    },
    saveConfig() {
      // 这里可以添加保存配置文件的逻辑
      // 例如，将 this.configContent 发送到后端保存
      //todo 向后端发送HTTP请求保存数据
      this.configDialogVisible = false; // 关闭配置文件对话框
    }
  }
};
</script>