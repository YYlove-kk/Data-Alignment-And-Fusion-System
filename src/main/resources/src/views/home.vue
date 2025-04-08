<template>
  <div class="content">

      <div class="refresh-settings">
        <div style="display: flex; align-items: center; justify-content: space-between;">
          <h3 style="font-size: 20px; color: #333;">设置轮询刷新时间</h3>
          <label>当前轮询刷新时间：{{ displayTime }}</label>
        </div>

        <label>月:
            <input type="number" v-model.number="months" placeholder="月" min="0" />
          </label>
          <label>天:
            <input type="number" v-model.number="days" placeholder="天" min="0" />
          </label>
          <label>小时:
            <input type="number" v-model.number="hours" placeholder="小时" min="0" />
          </label>
          <label>分钟:
            <input type="number" v-model.number="minutes" placeholder="分钟" min="0" />
          </label>
        <button @click="applyRefreshTime" style="width: 200px; margin-left: 100px">应用轮询时间</button>
      </div>



    <div class="chart-container">
      <div class="chart-item">
        <div class="selector-container">
        </div>
        <div id="chart-quantity" class="chart"></div>
      </div>
      <div class="chart-item">
        <div class="selector-container" style="display: flex; align-items: center; ">
          <label for="part-selector" style="margin-right: 10px; white-space: nowrap;">选择零件号：</label>
          <select id="part-selector" v-model="selectedPart" @change="updateSocChart" class="chart-select">
            <option v-for="part in chartData.categories" :key="part" :value="part">{{ part }}</option>
          </select>
        </div>

        <div id="chart-soc" class="chart"></div>
      </div>
      <div class="chart-item">
        <div class="selector-container"  style="display: flex; align-items: center; ">
          <label for="part-selector" style="margin-right: 10px; white-space: nowrap;">选择零件号：</label>
          <select v-model="selectedPartForTemperature" @change="updateTemperatureChart" class="chart-select">
            <option v-for="part in chartData.categories" :key="part" :value="part">{{ part }}</option>
          </select>
        </div>
        <div id="chart-temperature" class="chart"></div>
      </div>
      <div class="chart-item">
        <div class="selector-container">
          <div style="display: flex; align-items: center; ">
            <label for="part-selector" style="margin-right: 10px; white-space: nowrap;">选择类型：</label>
          <select v-model="selectedChartType" class="chart-select">
            <option value="version">WIL Version</option>
            <option value="script">BMS Script Version</option>
          </select>
          </div>
          <div style="display: flex; align-items: center; ">
            <label for="part-selector" style="margin-right: 10px; white-space: nowrap;">选择零件号：</label>
          <select v-model="selectedPartForConfig" @change="updateChart" class="chart-select">
            <option v-for="part in chartData.categories" :key="part" :value="part">{{ part }}</option>
          </select>
          </div>
        </div>
        <div v-if="selectedChartType === 'version'" id="chart-version" class="chart"></div>
        <div v-if="selectedChartType === 'script'" id="chart-script" class="chart"></div>
      </div>
    </div>
<!--    <list class="main-content" v-if="showList" :parent="this" :partNumber=showPart :socRange=socRange :showType=chartType :version=selectedVersion-->
<!--          :temperature=temperature :isShowDetailBtn=true></list>-->

    <exceptionList v-if="showExceptionList" :partNumber="showPart" :exceptionStatus="selectedExceptionStatus"></exceptionList>
  </div>
</template>


<style scoped>
.selector-container {
  height: 80px; /* 或根据最多选择条件的高度进行调整 */
}

.content {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: flex-start;
  width: 100%;
  height: 100%;
  padding: 20px;
  background-color: #f0f2f5;
}

.chart-container {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 20px;
  width: 100%;
  max-width: 1200px; /* Adjust based on your layout */
  margin-top: 20px;
}

.chart-item {
  background: #ffffff;
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  padding: 15px;
  display: flex;
  flex-direction: column;
}

.chart {
  height: 300px; /* Adjust height as needed */
  margin-top: 15px;
}

select {
  width: 100%;
  padding: 8px 12px;
  margin-bottom: 15px;
  border: 1px solid #d9d9d9;
  border-radius: 4px;
  background-color: #fff;
  box-shadow: inset 0 1px 3px rgb(0 0 0 / 0.1);
}

.back-button {
  padding: 10px 25px;
  margin-top: 30px;
  color: #fff;
  background-color: #1890ff;
  border: none;
  border-radius: 5px;
  cursor: pointer;
  transition: background 0.3s ease;

  &:hover {
    background-color: #40a9ff;
  }


  //轮询样式：

  .refresh-settings label {

  }

  input[type="number"] {
    width: 100%;
    padding: 8px;
    margin-top: 5px;
    border: 1px solid #ccc;
    border-radius: 5px;
  }

  button {
    width: 100%;
    padding: 10px;
    margin-top: 20px;
    background-color: #0066cc;
    color: white;
    font-size: 18px;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    transition: background-color 0.3s;
  }

  button:hover {
    background-color: #0056b3;
  }



}


@media (max-width: 768px) {
  .content {
    padding: 10px; /* 减少内边距 */
  }

  .chart-container {
    grid-template-columns: 1fr; /* 在移动端，图表以单列显示 */
    gap: 15px; /* 减少图表之间的间隙 */
  }

  .chart-item {
    padding: 10px; /* 减少内边距，使内容更适合移动屏幕 */
  }

  .selector-container {
    flex-direction: column; /* 选择器变为垂直排列 */
    gap: 5px; /* 增加选择器之间的间隙 */
    margin-bottom: 10px; /* 增加选择器容器和图表之间的间隙 */
  }

  .chart {
    height: 250px; /* 调整图表高度以适应移动端屏幕 */
  }

  .chart-select {
    width: auto; /* 调整选择框宽度以适应内容 */
  }

  .back-button {
    width: 100%; /* 使返回按钮宽度与屏幕宽度一致 */
    margin-top: 20px; /* 增加按钮顶部间隙 */
  }

  /* 如果有其他需要调整的样式，可以继续添加 */
}

</style>

<script>
import * as echarts from 'echarts';
import router from '@/router/router-static';
import list from '@/views/modules/BatteryModule/list.vue';
import exceptionList from '@/views/modules/ExceptionBatteryModule/list.vue';
import base from "@/utils/base";

export default {
  data() {
    return {
      selectedPart: '',
      selectedPartForTemperature: '',
      selectedPartForConfig: '',
      selectedChartType: 'version', // 新增状态，用于选择显示版本图还是脚本图
      selectedVersion:'',
      showPart:'',
      socRange: '',
      temperature: '',
      chartType:'',
      // showList: false,
      showExceptionList:false,
      selectedExceptionStatus: '', // 添加以保存选择的异常状态
      exceptionDataList: [], // 异常数据列表初始化
      // showChart: true,
      dataList:[],
      versionOrder:['V1.1.4', 'V2.0.0', 'V2.0.4', 'V2.0.8', 'V2.0.9'],
      scriptVersionOrder:['V10', 'V11', 'V12'],
      chartData: {
        // categories: ['24120503', '24120504', '24120149', '24120150', '24120443','24120444'],
        // quantity: [5, 20, 36, 10, 10, 20],
        // soc: [75, 80, 65, 90, 70, 60],
        // temperature: [22, 25, 24, 26, 23, 27],
        // version: [1, 2, 1, 3, 2, 3],
        categories: [],
        quantity: [],
        soc: [],
        temperature: [],
        version: [],
        socData: {
          // '24120503': [20, 30, 50, 70, 90],
          // '24120504': [15, 25, 35, 45, 55],
          // '24120149': [10, 20, 30, 40, 50],
          // '24120150': [25, 35, 45, 55, 65],
          // '24120443': [30, 40, 50, 60, 70],
          // '24120444': [35, 45, 55, 65, 75],
          // // 添加更多零件的SOC数据...
        },

        temperatureData: {
          // '24120503': [2, 5, 3, 6, 2],
          // '24120504': [1, 3, 2, 4, 5],
          // '24120149': [2, 4, 6, 8, 10],
          // '24120150': [3, 5, 7, 9, 11],
          // '24120443': [4, 6, 8, 10, 12],
          // '24120444': [5, 7, 9, 11, 13],
          // 添加更多零件的温度数据...
        },
        versionData: {
          // '24120503': {'V1.1.4': 10, 'V2.0.0': 12,'V2.0.4': 15, 'V2.0.8': 14,'V2.0.9': 2},
          // '24120504': {'V1.1.4': 8, 'V2.0.0': 14,'V2.0.4': 14, 'V2.0.8': 16,'V2.0.9': 4},
          // '24120149': {'V1.1.4': 8, 'V2.0.0': 13,'V2.0.4': 16, 'V2.0.8': 12,'V2.0.9': 1},
          // '24120150': {'V1.1.4': 6, 'V2.0.0': 12,'V2.0.4': 16, 'V2.0.8': 13,'V2.0.9': 5},
          // '24120443': {'V1.1.4': 9, 'V2.0.0': 13,'V2.0.4': 17, 'V2.0.8': 8,'V2.0.9': 6},
          // '24120444': {'V1.1.4': 11, 'V2.0.0': 10,'V2.0.4': 14, 'V2.0.8': 9,'V2.0.9': 3},
        },
        scriptVersionData: {
          // '24120503': {'V10': 20, 'V11': 15, 'V12': 25},
          // '24120504': {'V10': 18, 'V11': 22, 'V12': 30},
          // '24120149': {'V10': 25, 'V11': 20, 'V12': 35},
          // '24120150': {'V10': 15, 'V11': 18, 'V12': 28},
          // '24120443': {'V10': 22, 'V11': 26, 'V12': 32},
          // '24120444': {'V10': 30, 'V11': 25, 'V12': 20},
        },
      },
      refreshInterval: null, // 用于保存定时器 ID
      refreshTime: 30000, // T 时间间隔（毫秒），定期刷新页面时间(5分钟刷新一次)
      isLoadingData: false,

      months: 0,
      days: 0,
      hours: 0,
      minutes: 0,
      currentRefreshTime: base.getRefreshTime(), // 初始从base获取时间
      displayTime: '' // 格式化显示时间
    };
  },

  mounted() {
    this.fetchData();
    this.setupInterval();
    // this.init();
    // //异步数据加载等待
    // Promise.all([
    //   this.getFullDataList(),
    //   this.getExceptionDataList()
    // ]).then(() => {
    //   // 在数据加载完毕后更新图表
    //   this.updateChartData();
    // }).catch(error => {
    //   console.error("Error loading data:", error);
    // });

    // this.refreshInterval = setInterval(() => {
    //   this.getFullDataList(); // 定时获取数据
    //   this.getExceptionDataList();
    // }, this.refreshTime);
    this.formatDisplayTime();
  },
  beforeDestroy() {
    this.clearInterval();
  },

  deactivated() {
    this.clearInterval();
  },

  activated() {
    this.setupInterval();
  },


  watch: {
    selectedChartType(newVal, oldVal) {
      this.$nextTick(() => {
        this.selectedChartType = newVal;
        this.updateChart(newVal);
      });
    },
    selectedPart() {
      this.updateChart(this.selectedChartType);
    },
  },

  components:{
    list,
    exceptionList
  },

  methods: {
    init() {
      Promise.all([
        this.getFullDataList(),
        this.getExceptionDataList()
      ]).then(() => {
      }).catch(error => {
        console.error("Error loading data:", error);
      }).finally(() => {
        this.isLoadingData = false;
      });
    },

    fetchData() {
      if (this.isLoadingData) return;
      this.isLoadingData = true;

      Promise.all([
        this.getFullDataList(),
        this.getExceptionDataList()
      ]).then(() => {
        this.updateChartData();
      }).catch(error => {
        console.error("Error loading data:", error);
      }).finally(() => {
        this.isLoadingData = false;
      });
    },

    applyRefreshTime() {
      let totalMinutes = this.months * 43200 + this.days * 1440 + this.hours * 60 + this.minutes;
      base.setRefreshTime(totalMinutes);
      this.currentRefreshTime = base.getRefreshTime();
      this.formatDisplayTime();
      alert(`刷新时间已设置。`);
    },

    setupInterval() {
      // 清除现有的定时器
      if (this.refreshInterval) {
        clearInterval(this.refreshInterval);
      }
      // 基于最新的刷新时间设置新的定时器
      this.refreshInterval = setInterval(this.fetchData, base.getRefreshTime());
      console.log("New:  "+base.getRefreshTime());
    },
    resetInterval() {
      // 重新设置轮询定时器
      this.setupInterval();
    },


    clearInterval() {
      if (this.refreshInterval) clearInterval(this.refreshInterval);
    },

    // 格式化显示时间
    formatDisplayTime() {
      let totalMilliseconds = this.currentRefreshTime;
      let minutes = Math.floor(totalMilliseconds / 60000);
      let hours = Math.floor(minutes / 60);
      minutes = minutes % 60;
      let days = Math.floor(hours / 24);
      hours = hours % 24;
      let months = Math.floor(days / 30);
      days = days % 30;

      this.displayTime = `${months} 月 ${days} 天 ${hours} 小时 ${minutes} 分钟`;
    },

    async getFullDataList() {
      // console.log("GET FULL DATA:");
      try {
        const response = await this.$http.get("BatteryModule/infos");
        const data = response.data;

        if (data && data.status === 0) {
          this.dataList = data.data;
          if (this.dataList.length > 0) {
            const firstPartNumber = this.dataList[0].partNumber;
            this.selectedPart = firstPartNumber;
            this.selectedPartForTemperature = firstPartNumber;
            this.selectedPartForConfig = firstPartNumber;
          }
        } else {
          this.dataList = [];
        }
      } catch (error) {
        console.error("Failed to fetch data:", error);
      } finally {
        // console.error("Failed to fetch data:", error);
      }
    },
    async getExceptionDataList() {
      try {
        const response = await this.$http.get("BatteryModule/exceptionInfos");
        const data = response.data;
        if (data && data.status === 0) {
          this.exceptionDataList = data.data;
          // 根据需要进行数据处理...
        } else {
          this.exceptionDataList = [];
        }
      } catch (error) {
        console.error("Failed to fetch exception data:", error);
      }
    },

    updateChartData() {
      this.chartData.categories = [...new Set(this.dataList.map(item => item.partNumber))];
      this.initializeChartData();

      this.chartData.categories.forEach(partNumber => {
        const filteredData = this.dataList.filter(item => item.partNumber === partNumber);
        const exceptionData = this.exceptionDataList.filter(item => item.partNumber === partNumber);

        this.updateAccountData(partNumber, filteredData);

        // 对正常数据进行更新
        this.updateSocData(partNumber, filteredData);
        this.updateTemperatureData(partNumber, filteredData);

        // 更新版本数据和脚本版本数据不受异常数据影响
        this.updateVersionData(partNumber, filteredData);
        this.updateScriptVersionData(partNumber, filteredData);
      });

      // 在这里调用更新图表的方法，例如：
      this.updateAccountChart();
      this.updateSocChart();
      this.updateTemperatureChart();
      this.updateVersionChart();
      this.updateScriptChart();

    },

    initializeChartData() {
      this.chartData.socData = {};
      this.chartData.temperatureData = {};
      this.chartData.versionData = {};
      this.chartData.scriptVersionData = {};
      this.chartData.categories.forEach(partNumber => {
        this.chartData.versionData[partNumber] = {};
        this.chartData.scriptVersionData[partNumber] = {};
        this.versionOrder.forEach(version => {
          this.chartData.versionData[partNumber][version] = 0;
        });
        this.scriptVersionOrder.forEach(version => {
          this.chartData.scriptVersionData[partNumber][version] = 0;
        });
      });
    },

    updateAccountData(partNumber, filteredData) {
      // 根据partNumber计算对应的模组数量
      const count = filteredData.length;
      // 找到partNumber在categories中的索引，这个索引与quantity数组中的索引是对应的
      const index = this.chartData.categories.indexOf(partNumber);
      // 如果找到索引，就更新quantity数组中对应索引的值为count
      if(index !== -1) {
        this.$set(this.chartData.quantity, index, count);
      } else {
        // 如果找不到索引，说明有逻辑错误，因为在initializeChartData中应该已经初始化了categories和quantity
        console.error(`Part number ${partNumber} not found in categories.`);
      }
    },

    updateSocData(partNumber, filteredData) {
      const socSegments = [0, 20, 40, 60, 80, 100];
      this.chartData.socData[partNumber] = socSegments.slice(1).map((end, index) => {
        const start = socSegments[index];
        // 注意这里只过滤并计算正常数据的数量
        return filteredData.filter(item => item.moduleSoc >= start && item.moduleSoc <= end).length;
      });
    },

    updateTemperatureData(partNumber, filteredData) {
      const tempSegments = [-10, 0, 10, 20, 30, 40];
      this.chartData.temperatureData[partNumber] = tempSegments.slice(1).map((end, index) => {
        const start = tempSegments[index];
        // 同样只针对正常数据
        return filteredData.filter(item => item.moduleTemperature >= start && item.moduleTemperature <= end).length;
      });
    },



    updateVersionData(partNumber, filteredData) {
      this.chartData.versionData[partNumber] = {};

      // 初始化指定顺序的版本计数为0
      this.versionOrder.forEach(version => {
        this.chartData.versionData[partNumber][version] = 0;
      });

      // 对数据进行计数
      filteredData.forEach(item => {
        const version = item.wilVersion.trim().toUpperCase(); // 确保一致性
        if (version in this.chartData.versionData[partNumber]) {
          this.chartData.versionData[partNumber][version]++;
        }
      });
    },




    updateScriptVersionData(partNumber, filteredData) {
      this.chartData.scriptVersionData[partNumber] = {};

      // 初始化指定顺序的脚本版本计数为0
      this.scriptVersionOrder.forEach(version => {
        this.chartData.scriptVersionData[partNumber][version] = 0;
      });

      // 对数据进行计数
      filteredData.forEach(item => {
        const scriptVersion = item.scriptVersion.trim().toUpperCase(); // 确保一致性
        if (scriptVersion in this.chartData.scriptVersionData[partNumber]) {
          this.chartData.scriptVersionData[partNumber][scriptVersion]++;
        }
      });
    },




    updateAccountChart() {
      // 初始化数量和版本号图表
      this.createChart('chart-quantity', '数量 - 零件号', this.chartData.categories, this.chartData.quantity);
    },

    // 初始化SOC图表并添加点击事件
    // 初始化SOC图表并添加点击事件
    updateSocChart() {
      this.$nextTick(() => { // 确保DOM已更新
        const socData = [...this.chartData.socData[this.selectedPart]];
        // 假设异常数据中SOC异常类型为1
        const exceptionCount = this.exceptionDataList.filter(item => item.partNumber === this.selectedPart && item.exceptionStatus === '1').length;
        // 替换第一个柱状图数据为异常数据计数
        socData[0] = exceptionCount;

        const categories = ['0%~20%', '20%~40%', '40%~60%', '60%~80%', '80%~100%'];

        const chartElement = document.getElementById('chart-soc');
        if (chartElement) { // 检查元素是否存在
          const chart = echarts.init(chartElement);
          const option = {
            title: { text: `数量 - SOC - ${this.selectedPart}` },
            tooltip: {},
            xAxis: {
              type: 'category',
              data: categories,
            },
            yAxis: {
              type: 'value',
            },
            series: [{
              name: 'SOC',
              type: 'bar',
              data: socData,
              itemStyle: {
                // 指定柱状图颜色为绿色
                color: (params) => {
                  // 仅将第一个柱状图颜色设置为红色
                  return params.dataIndex === 0 ? 'rgb(229,4,56)' : 'rgba(81,70,236,0.8)';
                }
              }
            }],
          };
          chart.setOption(option);

          // 添加点击事件监听
          chart.on('click', (params) => {
            if (params.dataIndex === 0) { // 假设点击第一个柱子跳转到异常列表
              this.showExceptionList = true;
              this.selectedExceptionStatus = 'SOC异常'; // SOC异常
              this.showPart = this.selectedPart;
              this.navigateToList(); // 更新此函数以处理异常列表的导航
            } else {
              this.socRange = params.name;
              this.showPart = this.selectedPart;
              this.navigateToList();
            }
          });
        } else {
          console.error('Element #chart-soc not found');
        }
      });
    },


    // 更新温度图表
    updateTemperatureChart() {
      this.$nextTick(() => { // 确保DOM已更新
        const temperatureData = [...this.chartData.temperatureData[this.selectedPartForTemperature]];
        // 假设温度异常类型为2
        temperatureData[0] = this.exceptionDataList.filter(item => item.partNumber === this.selectedPartForTemperature && item.exceptionStatus === '2' && item.moduleTemperature >= -10 && item.moduleTemperature <= 0).length;
        temperatureData[4] = this.exceptionDataList.filter(item => item.partNumber === this.selectedPartForTemperature && item.exceptionStatus === '2' && item.moduleTemperature >= 30 && item.moduleTemperature <= 40).length;

        const categories = ['-10°~0°', '0°~10°', '10°~20°', '20°~30°', '30°~40°'];

        const chartElement = document.getElementById('chart-temperature');
        if (chartElement) { // 检查元素是否存在
          const chart = echarts.init(chartElement);
          const option = {
            title: { text: `数量 - 温度 - ${this.selectedPartForTemperature}` },
            tooltip: {},
            xAxis: {
              type: 'category',
              data: categories,
            },
            yAxis: {
              type: 'value',
            },
            series: [{
              name: '温度',
              type: 'bar',
              data: temperatureData,
              itemStyle: {
                // 指定柱状图颜色为绿色
                color: (params) => {
                  return params.dataIndex === 0 || params.dataIndex === 4 ? 'rgb(229,4,56)' : 'rgba(81,70,236,0.8)';
                }
              }
            }],
          };
          chart.setOption(option);

          // 添加点击事件监听
          chart.on('click', (params) => {
            if (params.dataIndex === 0 || params.dataIndex === 4) { // 假设点击异常柱子跳转到异常列表
              this.showExceptionList = true;
              this.exceptionStatus = '温度异常';
              this.showPart = this.selectedPartForTemperature;
              this.navigateToList(); // 处理异常列表的导航
            } else {
              this.temperature = params.name;
              this.showPart = this.selectedPartForTemperature;
              this.navigateToList();
            }
          });
        } else {
          console.error('Element #chart-temperature not found');
        }
      });
    },


    updateChart(chartType) {
      if (chartType === 'version') {
        this.updateVersionChart();
      } else if (chartType === 'script') {
        this.updateScriptChart();
      }
    },

    // 更新版本图表
    updateVersionChart() {
      this.$nextTick(() => { // 确保DOM更新完成后执行
        const versionData = this.chartData.versionData[this.selectedPartForConfig];
        const categories = ['V1.1.4', 'V2.0.0', 'V2.0.4', "V2.0.8", "V2.0.9"];
        const data = Object.values(versionData); // 获取版本的所有值作为数据

        const chartElement = document.getElementById('chart-version');
        if (chartElement) { // 检查元素是否存在
          const chart = echarts.init(chartElement);
          const option = {
            title: { text: `数量 - 版本 - ${this.selectedPartForConfig}` },
            tooltip: {},
            xAxis: {
              type: 'category',
              data: categories,
            },
            yAxis: {
              type: 'value',
            },
            series: [{
              name: '版本号',
              type: 'bar',
              data: data,
              itemStyle: {
                // 指定柱状图颜色为绿色
                color: 'rgba(81,70,236,0.8)'
              }
              // 配置项，例如颜色，可以根据需要添加
            }],
          };
          chart.setOption(option);

          // 添加点击事件监听
          chart.on('click', (params) => {
            this.showPart = this.selectedPartForConfig;
            this.chartType = this.selectedChartType;
            this.selectedVersion = params.name;
            this.navigateToList(); // 跳转到列表页
          });
        } else {
          console.error('Element #chart-version not found');
        }
      });
    },


    updateScriptChart() {
      this.$nextTick(() => {
        if (this.selectedChartType !== 'script') {
          return;
        }

        const scriptData = this.chartData.scriptVersionData[this.selectedPartForConfig];
        const categories = ['V10', 'V11', 'V12'];
        const data = Object.values(scriptData);

        const chartElement = document.getElementById('chart-script');
        if (chartElement) {
          const chart = echarts.init(chartElement);
          const option = {
            title: { text: `数量 - 版本 - ${this.selectedPartForConfig}` },
            tooltip: {},
            xAxis: { type: 'category', data: categories },
            yAxis: { type: 'value' },
            series: [{
              name: '脚本号',
              type: 'bar',
              data: data,
              itemStyle: { color: 'rgba(81,70,236,0.8)' }
            }],
          };
          chart.setOption(option);

          chart.on('click', (params) => {
            this.showPart = this.selectedPartForConfig;
            this.chartType = this.selectedChartType;
            this.selectedVersion = params.name;
            this.navigateToList();
          });
        } else {
          console.error('Element #chart-script not found');
        }
      });
    },



    createChart(elementId, title, categories, data) {
      const chart = echarts.init(document.getElementById(elementId));
      const option = {
        title: { text: title },
        tooltip: {},
        xAxis: { type: 'category', data: categories },
        yAxis: { type: 'value' },
        series: [{ name: title, type: 'bar', data: data ,itemStyle: {
            // 指定柱状图颜色为绿色
            color: 'rgba(81,70,236,0.8)'
          }}]
      };
      chart.setOption(option);
    },

    navigateToList() {
      if (this.showExceptionList) {
        console.log("GOTO EXC" + this.showPart + "   EXC:   " + this.selectedExceptionStatus);
        // 跳转到异常数据列表，假设路由已正确设置
        router.replace({
          name: '异常模组状态',
          params: {
            partNumber: this.showPart,
            exceptionStatus: this.selectedExceptionStatus,
          }
        });
      }else{
        // 使用router.replace来进行页面跳转，并传递参数
        router.replace({
          name: '模组状态', // 或者是 'ExceptionBatteryModule' 根据您的需求选择正确的路由名称
          params: {
            partNumber: this.showPart,
            socRange: this.socRange,
            temperature: this.temperature,
            showType: this.chartType,
            version: this.selectedVersion,
            isShowDetailBtn:true,
            // 可以继续添加其他需要传递的参数
          }
        });
      }
    },
  }
};
</script>

