<template>
  <div class="knowledge-graph">
    <!-- 页面主体 -->
    <div class="container">
      <!-- 左侧: 构建 & 融合配置 -->
      <div class="left-panel">
        <h2>图谱构建 & 融合配置</h2>

        <!-- 图谱构建模块 -->
        <div class="panel-section">
          <h3>知识图谱构建</h3>
          <div class="form-group">
            <label for="buildDataSource">选择对齐后的数据</label>
            <select 
              id="buildDataSource" 
              v-model="buildConfig.dataSource"
            >
              <option value="">选择可用的数据集</option>
              <option value="alignment1">对齐结果 #1</option>
              <option value="alignment2">对齐结果 #2</option>
            </select>
          </div>

          <div class="form-group">
            <label for="graphBuildMode">构建模式</label>
            <select 
              id="graphBuildMode" 
              v-model="buildConfig.buildMode"
            >
              <option value="single">单模态</option>
              <option value="multi">多模态</option>
            </select>
          </div>

          <button 
            class="action-btn"
            @click="startBuild"
            :disabled="!canStartBuild"
          >
            开始构建图谱
          </button>
        </div>

        <!-- 图谱融合模块 -->
        <div class="panel-section">
          <h3>知识图谱融合</h3>
          <div class="form-group">
            <label for="existingGraphSelect">融合到已有图谱</label>
            <select 
              id="existingGraphSelect" 
              v-model="fusionConfig.targetGraph"
            >
              <option value="">主知识图谱 #1</option>
              <option value="graph2">主知识图谱 #2</option>
            </select>
          </div>

          <div class="form-group">
            <label for="fusionStrategy">融合策略</label>
            <select 
              id="fusionStrategy" 
              v-model="fusionConfig.strategy"
            >
              <option value="gnn">GNN多头注意力</option>
              <option value="heuristic">启发式合并</option>
              <option value="rules">规则驱动</option>
            </select>
          </div>

          <button 
            class="action-btn"
            @click="startFusion"
            :disabled="!canStartFusion"
          >
            开始融合
          </button>
        </div>
      </div>

      <!-- 中间: 可视化 + 结果显示 -->
      <div class="main-content">
        <h2>知识图谱构建 & 融合结果</h2>

        <!-- Tab切换 -->
        <div class="tab-header">
          <button 
            :class="{ active: activeTab === 'build' }"
            @click="activeTab = 'build'"
          >
            构建结果
          </button>
          <button 
            :class="{ active: activeTab === 'fusion' }"
            @click="activeTab = 'fusion'"
          >
            融合结果
          </button>
        </div>

        <!-- 构建结果面板 -->
        <div 
          class="tab-content" 
          v-show="activeTab === 'build'"
        >
          <div class="graph-visual">
            <div v-if="!isBuilding" class="placeholder">
              这里显示图谱构建后的局部可视化（或力导向图）
            </div>
            <div v-else class="visualization">
              <div class="loading">正在生成可视化...</div>
            </div>
          </div>
          <table class="result-table">
            <thead>
              <tr>
                <th>节点名称</th>
                <th>模态</th>
                <th>属性数</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(node, index) in buildResults" :key="index">
                <td>{{ node.name }}</td>
                <td>{{ node.modality }}</td>
                <td>{{ node.attributeCount }}</td>
              </tr>
            </tbody>
          </table>
        </div>

        <!-- 融合结果面板 -->
        <div 
          class="tab-content" 
          v-show="activeTab === 'fusion'"
        >
          <div class="graph-visual">
            <div v-if="!isFusing" class="placeholder">
              这里展示融合后，与主知识图谱合并的可视化
            </div>
            <div v-else class="visualization">
              <div class="loading">正在生成可视化...</div>
            </div>
          </div>
          <table class="result-table">
            <thead>
              <tr>
                <th>节点名称</th>
                <th>所属图谱</th>
                <th>合并方式</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(node, index) in fusionResults" :key="index">
                <td>{{ node.name }}</td>
                <td>{{ node.graph }}</td>
                <td>{{ node.mergeMethod }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue';

// 构建配置
const buildConfig = ref({
  dataSource: '',
  buildMode: 'single'
});

// 融合配置
const fusionConfig = ref({
  targetGraph: '',
  strategy: 'gnn'
});

// 构建状态
const isBuilding = ref(false);

// 融合状态
const isFusing = ref(false);

// 当前激活的标签页
const activeTab = ref('build');

// 构建结果
const buildResults = ref([
  {
    name: 'Patient_B200',
    modality: '患者',
    attributeCount: 5
  }
]);

// 融合结果
const fusionResults = ref([
  {
    name: 'Disease_Heart',
    graph: '主图谱 #1',
    mergeMethod: 'GNN聚合(相似度0.8)'
  }
]);

// 计算属性：是否可以开始构建
const canStartBuild = computed(() => {
  return buildConfig.value.dataSource !== '';
});

// 计算属性：是否可以开始融合
const canStartFusion = computed(() => {
  return fusionConfig.value.targetGraph !== '' && 
         fusionConfig.value.strategy !== '';
});

// 开始构建
const startBuild = async () => {
  if (!canStartBuild.value) return;
  
  isBuilding.value = true;
  
  try {
    // 模拟构建过程
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // 更新构建结果
    buildResults.value = [
      {
        name: 'Patient_B200',
        modality: '患者',
        attributeCount: 5
      },
      {
        name: 'Disease_Heart',
        modality: '疾病',
        attributeCount: 3
      }
    ];
  } catch (error) {
    console.error('构建过程出错:', error);
  } finally {
    isBuilding.value = false;
  }
};

// 开始融合
const startFusion = async () => {
  if (!canStartFusion.value) return;
  
  isFusing.value = true;
  
  try {
    // 模拟融合过程
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // 更新融合结果
    fusionResults.value = [
      {
        name: 'Disease_Heart',
        graph: '主图谱 #1',
        mergeMethod: 'GNN聚合(相似度0.8)'
      },
      {
        name: 'Patient_B201',
        graph: '主图谱 #2',
        mergeMethod: '启发式合并'
      }
    ];
  } catch (error) {
    console.error('融合过程出错:', error);
  } finally {
    isFusing.value = false;
  }
};
</script>

<style lang="scss" scoped>
.knowledge-graph {
  .container {
    margin: 1rem 2rem;
    display: flex;
    gap: 1rem;
  }

  .left-panel {
    flex: 0 0 300px;
    background-color: #fff;
    border-radius: 6px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    padding: 1rem;

    h2 {
      font-size: 1rem;
      margin-bottom: 0.8rem;
      color: #007bff;
    }

    .panel-section {
      margin-bottom: 1.5rem;
      background-color: #f0f3f5;
      padding: 0.8rem;
      border-radius: 4px;

      h3 {
        font-size: 1rem;
        color: #333;
        margin-bottom: 0.5rem;
      }

      .form-group {
        margin-bottom: 0.8rem;

        label {
          display: block;
          margin-bottom: 0.3rem;
          font-weight: bold;
          color: #444;
        }

        select, input {
          width: 100%;
          padding: 0.4rem;
          margin-bottom: 0.8rem;
          border: 1px solid #ccc;
          border-radius: 4px;
        }
      }

      .action-btn {
        width: 100%;
        padding: 0.5rem;
        background-color: #28a745;
        color: #fff;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        font-size: 0.95rem;

        &:hover {
          background-color: #218838;
        }

        &:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }
      }
    }
  }

  .main-content {
    flex: 1;
    background-color: #fff;
    border-radius: 6px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    padding: 1rem;

    h2 {
      font-size: 1rem;
      margin-bottom: 0.8rem;
      color: #007bff;
    }

    .tab-header {
      display: flex;
      gap: 1rem;
      margin-bottom: 1rem;

      button {
        padding: 0.4rem 0.8rem;
        background-color: #e9ecef;
        color: #333;
        border: 1px solid #ccc;
        border-radius: 4px;
        cursor: pointer;

        &.active {
          background-color: #007bff;
          color: #fff;
          border-color: #007bff;
        }
      }
    }

    .graph-visual {
      border: 1px dashed #ccc;
      height: 350px;
      margin-bottom: 1rem;
      display: flex;
      justify-content: center;
      align-items: center;
      color: #999;
      border-radius: 4px;

      .placeholder, .loading {
        text-align: center;
      }
    }

    .result-table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 1rem;

      th, td {
        border: 1px solid #ddd;
        padding: 0.6rem;
        text-align: left;
      }

      th {
        background-color: #f7f7f7;
      }
    }
  }
}
</style>
