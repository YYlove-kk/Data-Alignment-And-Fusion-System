<template>
  <div class="data-alignment">
    <!-- 主体容器 -->
    <div class="container">
      <!-- 左侧配置: 对齐参数 & 执行操作 -->
      <div class="sidebar">
        <h2>对齐配置</h2>

        <!-- 时序对齐设置 -->
        <div class="form-group">
          <label for="timeAlignMethod">时序对齐算法</label>
          <select 
            id="timeAlignMethod" 
            v-model="alignmentConfig.timeAlignMethod"
          >
            <option value="dtw">DTW</option>
            <option value="transformer">自适应时间编码器</option>
            <option value="other">其他</option>
          </select>
        </div>

        <div class="form-group">
          <label for="timeGranularity">时间粒度(小时/天)</label>
          <input 
            type="number" 
            id="timeGranularity" 
            v-model="alignmentConfig.timeGranularity"
            placeholder="例如24" 
          />
        </div>

        <!-- 语义对齐设置 -->
        <div class="form-group">
          <label for="semanticModel">语义对齐模型</label>
          <select 
            id="semanticModel" 
            v-model="alignmentConfig.semanticModel"
          >
            <option value="clip">CLIP</option>
            <option value="bertcnn">BERT + CNN</option>
            <option value="custom">自定义对比学习</option>
          </select>
        </div>

        <div class="form-group">
          <label for="similarityThreshold">相似度阈值</label>
          <input 
            type="text" 
            id="similarityThreshold" 
            v-model="alignmentConfig.similarityThreshold"
            placeholder="如0.7" 
          />
        </div>

        <button 
          class="start-align-btn"
          @click="startAlignment"
          :disabled="!canStartAlignment"
        >
          开始对齐
        </button>
      </div>

      <!-- 中间结果区: 可视化 + 对齐状态 -->
      <div class="main-content">
        <h2>对齐结果概览</h2>
        <!-- 指标卡片 -->
        <div class="metrics">
          <div class="metric-card">
            <h4>时序误差</h4>
            <p>{{ metrics.timeError || '--' }}</p>
          </div>
          <div class="metric-card">
            <h4>语义准确率</h4>
            <p>{{ metrics.semanticAccuracy || '--' }}</p>
          </div>
          <div class="metric-card">
            <h4>对齐覆盖数</h4>
            <p>{{ metrics.alignmentCount || '--' }}</p>
          </div>
        </div>

        <!-- 对齐可视化 -->
        <div class="align-visual">
          <div v-if="!isAligning" class="placeholder">
            时序/语义对齐图可在此处渲染（热力图、Attention可视化等）
          </div>
          <div v-else class="visualization">
            <!-- 这里可以添加实际的可视化组件 -->
            <div class="loading">正在生成可视化...</div>
          </div>
        </div>

        <!-- 对齐详情表 -->
        <table class="alignment-table">
          <thead>
            <tr>
              <th>文本ID/描述</th>
              <th>图像ID/描述</th>
              <th>时序对齐</th>
              <th>语义相似度</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(item, index) in alignmentResults" :key="index">
              <td>{{ item.textId }}</td>
              <td>{{ item.imageId }}</td>
              <td>{{ item.timeAlignment }}</td>
              <td>{{ item.similarity }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue';

// 对齐配置
const alignmentConfig = ref({
  timeAlignMethod: 'dtw',
  timeGranularity: 24,
  semanticModel: 'clip',
  similarityThreshold: 0.7
});

// 对齐状态
const isAligning = ref(false);

// 对齐指标
const metrics = ref({
  timeError: null,
  semanticAccuracy: null,
  alignmentCount: null
});

// 对齐结果
const alignmentResults = ref([
  {
    textId: 'ExampleText_1',
    imageId: 'Img_2024_05_10',
    timeAlignment: '时间差2h',
    similarity: 0.72
  }
]);

// 计算属性：是否可以开始对齐
const canStartAlignment = computed(() => {
  return alignmentConfig.value.timeGranularity > 0 && 
         alignmentConfig.value.similarityThreshold > 0;
});

// 开始对齐
const startAlignment = async () => {
  if (!canStartAlignment.value) return;
  
  isAligning.value = true;
  
  try {
    // 模拟对齐过程
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // 更新指标
    metrics.value = {
      timeError: '0.5h',
      semanticAccuracy: '85%',
      alignmentCount: 156
    };
    
    // 更新结果
    alignmentResults.value = [
      {
        textId: 'ExampleText_1',
        imageId: 'Img_2024_05_10',
        timeAlignment: '时间差2h',
        similarity: 0.72
      },
      {
        textId: 'ExampleText_2',
        imageId: 'Img_2024_05_11',
        timeAlignment: '时间差1h',
        similarity: 0.85
      }
    ];
  } catch (error) {
    console.error('对齐过程出错:', error);
  } finally {
    isAligning.value = false;
  }
};
</script>

<style lang="scss" scoped>
.data-alignment {
  .container {
    display: flex;
    margin: 1rem 2rem;
    min-height: calc(100vh - 80px);
  }

  .sidebar {
    flex: 0 0 280px;
    background-color: #fff;
    border-radius: 6px;
    margin-right: 1rem;
    padding: 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);

    h2 {
      font-size: 1rem;
      margin-bottom: 1rem;
      color: #007bff;
    }

    .form-group {
      margin-bottom: 1rem;

      label {
        display: block;
        margin: 0.5rem 0 0.2rem;
        font-weight: bold;
        color: #444;
      }

      select, input {
        width: 100%;
        padding: 0.4rem;
        margin-bottom: 0.7rem;
        border: 1px solid #ccc;
        border-radius: 4px;
      }
    }

    .start-align-btn {
      width: 100%;
      padding: 0.5rem;
      background-color: #28a745;
      color: #fff;
      font-size: 1rem;
      border: none;
      border-radius: 4px;
      cursor: pointer;

      &:hover {
        background-color: #218838;
      }

      &:disabled {
        opacity: 0.5;
        cursor: not-allowed;
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

    .metrics {
      display: flex;
      gap: 1rem;
      margin-bottom: 1rem;

      .metric-card {
        flex: 1;
        background-color: #f0f3f5;
        border-radius: 4px;
        padding: 0.8rem;
        text-align: center;

        h4 {
          font-weight: normal;
          color: #777;
          margin-bottom: 0.4rem;
        }

        p {
          font-size: 1.2rem;
          color: #333;
          font-weight: bold;
        }
      }
    }

    .align-visual {
      border: 1px dashed #ccc;
      height: 300px;
      margin-bottom: 1rem;
      display: flex;
      justify-content: center;
      align-items: center;
      color: #999;
      border-radius: 4px;

      .placeholder {
        text-align: center;
      }

      .visualization {
        width: 100%;
        height: 100%;
        display: flex;
        justify-content: center;
        align-items: center;
      }
    }

    .alignment-table {
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
