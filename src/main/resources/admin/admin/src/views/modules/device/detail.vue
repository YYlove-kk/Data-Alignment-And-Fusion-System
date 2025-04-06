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

.single-col-form-item, .chart-form-item {
  padding: 15px;
  background-color: #ffffff;
  border-radius: 8px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.05);
}

.input, .single-input {
  border-radius: 4px;
}

.btn {
  text-align: right;
  margin-top: 20px;
}
</style>

<template>
  <div class="addEdit-block">
    <el-form
        ref="ruleForm"
        :model="ruleForm"
        label-width="100px"
        class="detail-form-content"
    >
      <el-row :gutter="20">
        <el-form-item label="设备名称">
          <el-input
              v-model="ruleForm.deviceName"
              placeholder="设备名称"
              readonly
          />
        </el-form-item>

        <el-form-item label="设备区号">
          <el-input
              v-model="ruleForm.deviceArea"
              placeholder="设备区号"
              readonly
          />
        </el-form-item>

        <el-form-item label="设备内存">
          <el-input
              v-model="ruleForm.deviceMemory"
              placeholder="设备内存"
              readonly
          />
        </el-form-item>

        <el-form-item label="设备电量">
          <el-input
              v-model="ruleForm.deviceSoc"
              placeholder="设备电量"
              readonly
          />
        </el-form-item>

        <el-form-item label="设备上传下载速率">
          <el-input
              v-model="ruleForm.deviceLoadBytes"
              placeholder="设备上传下载速率"
              readonly
          />
        </el-form-item>

        <el-form-item label="设备开机状态" class="single-col-form-item">
          <el-input
              v-model="ruleForm.deviceStatus"
              placeholder="设备开机状态"
              readonly
          />
        </el-form-item>

        <el-form-item label="设备监控模组数" class="single-col-form-item">
          <el-input v-model="ruleForm.deviceModuleAccount" placeholder="设备监控模组数" readonly />
        </el-form-item>

        <el-form-item label="添加时间" class="single-col-form-item">
          <el-input v-model="ruleForm.deviceAddtime" placeholder="添加时间" readonly />
        </el-form-item>

      </el-row>

      <el-form-item>
        <el-button class="btn-close-back" @click="back()">返回</el-button>
      </el-form-item>

    </el-form>
  </div>
</template>


<script>
// 数字，邮件，手机，url，身份证校验
import { isNumber,isIntNumer,isEmail,isPhone, isMobile,isURL,checkIdCard } from "@/utils/validate";
import * as echarts from 'echarts';
export default {
  props: ["parent",'detailData'],
  data() {
    return {
    // socData: [3.5,2,1,3.1,3.2,3.5,4.0,3.1,3.2,3.5],
    socData: [3.5,2,1,3.1,3.2,3.5,4.0,3.1,3.2,3.5,3.4,3.3,3.0,2.8],
    FPA: "V1.X.X",
    version: "V1.0.0",
    BMS_Script_Version: "V10",
    config:"XXX",
    chartInstance: null,
    configDialogVisible: false, // 配置文件对话框可见性
    configContent: '此处显示配置文件',
	  addEditForm: {"btnSaveFontColor":"#fff","selectFontSize":"14px","btnCancelBorderColor":"#DCDFE6","inputBorderRadius":"4px","inputFontSize":"14px","textareaBgColor":"#fff","btnSaveFontSize":"14px","textareaBorderRadius":"4px","uploadBgColor":"#fff","textareaBorderStyle":"solid","btnCancelWidth":"88px","textareaHeight":"120px","dateBgColor":"#fff","btnSaveBorderRadius":"4px","uploadLableFontSize":"14px","textareaBorderWidth":"1px","inputLableColor":"#606266","addEditBoxColor":"#fff","dateIconFontSize":"14px","btnSaveBgColor":"rgba(117, 113, 249, 1)","uploadIconFontColor":"#8c939d","textareaBorderColor":"#DCDFE6","btnCancelBgColor":"#ecf5ff","selectLableColor":"#606266","btnSaveBorderStyle":"solid","dateBorderWidth":"1px","dateLableFontSize":"14px","dateBorderRadius":"4px","btnCancelBorderStyle":"solid","selectLableFontSize":"14px","selectBorderStyle":"solid","selectIconFontColor":"#C0C4CC","btnCancelHeight":"44px","inputHeight":"40px","btnCancelFontColor":"#606266","dateBorderColor":"#DCDFE6","dateIconFontColor":"#C0C4CC","uploadBorderStyle":"solid","dateBorderStyle":"solid","dateLableColor":"#606266","dateFontSize":"14px","inputBorderWidth":"1px","uploadIconFontSize":"28px","selectHeight":"40px","inputFontColor":"#606266","uploadHeight":"148px","textareaLableColor":"#606266","textareaLableFontSize":"14px","btnCancelFontSize":"14px","inputBorderStyle":"solid","btnCancelBorderRadius":"4px","inputBgColor":"#fff","inputLableFontSize":"14px","uploadLableColor":"#606266","uploadBorderRadius":"4px","btnSaveHeight":"44px","selectBgColor":"#fff","btnSaveWidth":"88px","selectIconFontSize":"14px","dateHeight":"40px","selectBorderColor":"#DCDFE6","inputBorderColor":"#DCDFE6","uploadBorderColor":"#DCDFE6","textareaFontColor":"#606266","selectBorderWidth":"1px","dateFontColor":"#606266","btnCancelBorderWidth":"1px","uploadBorderWidth":"1px","textareaFontSize":"14px","selectBorderRadius":"4px","selectFontColor":"#606266","btnSaveBorderColor":"rgba(117, 113, 249, 1)","btnSaveBorderWidth":"1px"},
      id: '',
      type: '',
          ruleForm: {
              deviceName: "xxx",
              deviceArea: "xxx",
              deviceMemory: "xxx",
              deviceSoc: "xxx", // 使用键对应的值
              deviceLoadBytes: "xxx",
              deviceStatus: 'xxx',
              deviceModuleAccount: 'xxx',
              deviceAddtime: 'xx-xx-xx-xx',
          },
    };
  },

  watch: {
    detailData: {
      immediate: true,
      handler(newVal) {
        if (newVal) {
          this.ruleForm = { // 直接使用传递的数据初始化ruleForm
            deviceName: newVal.deviceName,
            deviceArea: newVal.deviceArea,
            deviceMemory: newVal.deviceMemory,
            deviceSoc: newVal.deviceSoc,
            deviceLoadBytes: newVal.deviceLoadBytes,
            deviceStatus: newVal.deviceStatus,
            deviceModuleAccount: newVal.deviceModuleAccount,
            deviceAddtime: newVal.deviceAddtime,
          };
          // 调用初始化图表等其他方法
        }
      }
    }
  },

  mounted() {
    // this.getSocData();
    this.initChart();
    this.fetchFPA(); // 在组件创建时获取 FPA 数据
    this.fetchVersion(); // 在组件创建时获取 Version 数据
    this.fetchConfig(); // 在组件创建时获取 Config 数据
  },
  computed: {},
  created() {
	this.addEditStyleChange()
	this.addEditUploadStyleChange()
  },
  methods: {
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
      // 获取模组信息
      this.$http({
        url: `${this.$storage.get('sessionTable')}/session`,
        method: "get"
      }).then(({ data }) => {
        if (data && data.code === 0) {
          var json = data.data;
        }else {
          this.$message.error(data.msg);
        }
      });
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
    onSubmit() {
        this.$refs["ruleForm"].validate(valid => {
        if (valid) {
          this.$http({
            url: `BatteryModuleInfo/${!this.ruleForm.id ? "save" : "update"}`,
            method: "post",
            data: this.ruleForm
          }).then(({ data }) => {
            if (data && data.code === 0) {
              this.$message({
                message: "操作成功",
                type: "success",
                duration: 1500,
                onClose: () => {
                  this.parent.showFlag = true;
                  this.parent.showDetail = false;
                  this.parent.search();
                  this.parent.contentStyleChange();
                }
              });
            } else {
              this.$message.error(data.msg);
            }
          });
        }
      });
    },
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
    tupianUploadChange(fileUrls) {
      this.ruleForm.tupian = fileUrls;
      this.addEditUploadStyleChange()
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

   //获取模组SOC曲线
    getSocData() {
      // 示例URL，根据实际情况替换
      const url = 'BatteryModuleInfo/soc/id';
      console.log("getSoc");
      this.$http.get(url).then(response => {
        const data = response.data;
        if(data && data.code === 0) {
          // 假设data.data是一个数组，包含多个电池的SOC值
          this.socData = data.data;
          this.initChart();
        } else {
          console.error('Failed to fetch SOC data');
        }
      }).catch(error => {
        console.error('Error fetching SOC data:', error);
      });
    },
    // initChart() {
    //   this.chartInstance = echarts.init(this.$refs.socChart);
    //   const options = {
    //     tooltip: {},
    //     xAxis: {
    //       type: 'category',
    //       data: this.socData.map((item, index) => `电池${index + 1}`)
    //     },
    //     yAxis: {
    //       type: 'value'
    //     },
    //     series: [{
    //       data: this.socData,
    //       type: 'bar'
    //     }]
    //   };
    //
    //   this.chartInstance.setOption(options);
    // }
    initChart() {
      // 基于准备好的dom，初始化echarts实例
      const myChart = echarts.init(this.$refs.socChart);
      // 计算柱子的宽度
      const columnWidth = Math.max(50 / this.socData.length, 5);

      // 指定图表的配置项和数据
      const option = {
        tooltip: {},
        xAxis: {
          type: 'category',
          // data: ['电芯1', '电芯2', '电芯3', '电芯4','电芯5', '电芯6', '电芯7', '电芯8'],
          data: ['电芯1', '电芯2', '电芯3', '电芯4','电芯5', '电芯6', '电芯7', '电芯8','电芯9','电芯10','电芯11','电芯12'],
        },
        yAxis: {
          type: 'value',
          max: 5, // 设置y轴最大值为100
        },
        series: [
          {
            data: this.socData,
            type: 'bar',
            barWidth: columnWidth, // 设置柱子的宽度
            itemStyle: {
              normal: {
                color: function(params) {
                  // 低于30的柱子颜色为红色，其他为绿色
                  return params.value < 2.0 || params.value >= 4.0 ? 'red' : 'green';
                },
              },
            },
            showBackground: true,
            backgroundStyle: {
              color: 'rgba(180, 180, 180, 0.2)',
            },
          },
        ],
      };

      // 使用刚指定的配置项和数据显示图表。
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