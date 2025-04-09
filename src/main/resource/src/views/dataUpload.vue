<template>
  <div class="data-upload">
    <!-- 主体 -->
    <div class="container">
      <!-- 顶部介绍区 -->
      <div class="upload-header">
        <h2>数据上传</h2>
        <p>支持上传文件类型：CSV、Excel、JSON、医养康影像等。请确保字段中包含患者标识、机构信息等必需项，以便后续数据对齐与融合。文件大小限制等可在系统设置中查看。</p>
      </div>

      <!-- 左右分栏: 上传操作 / 已上传列表 -->
      <div class="upload-container">
        <!-- 左侧：上传操作 -->
        <div class="left-panel">
          <h3>上传文件</h3>
          <div 
            class="drag-area"
            @dragover.prevent
            @drop.prevent="handleDrop"
            @click="triggerFileInput"
          >
            <p>点击或拖拽文件到此处</p>
            <input 
              type="file" 
              ref="fileInput"
              style="display: none"
              @change="handleFileChange"
            />
          </div>

          <div class="info-field">
            <label for="dataSourceName">数据源名称</label>
            <input 
              type="text" 
              id="dataSourceName" 
              v-model="formData.dataSourceName"
              placeholder="例: 广医医院2024-Q1影像数据" 
            />
          </div>
          <div class="info-field">
            <label for="modalType">模态类型</label>
            <select id="modalType" v-model="formData.modalType">
              <option value="text">文本</option>
              <option value="image">图像</option>
              <option value="timeSeries">时序</option>
              <option value="other">其他</option>
            </select>
          </div>
          <div class="info-field">
            <label for="orgName">机构名称/备注</label>
            <input 
              type="text" 
              id="orgName" 
              v-model="formData.orgName"
              placeholder="例: 广东省人民医院" 
            />
          </div>

          <div class="upload-actions">
            <button @click="handleUpload" :disabled="!canUpload">开始上传</button>
            <button @click="handleCancel" style="background-color:#6c757d;">取消</button>
          </div>

          <div class="progress-area">
            <p>当前上传进度: {{ uploadProgress }}%</p>
          </div>
        </div>

        <!-- 右侧：已上传数据列表 -->
        <div class="right-panel">
          <h3>已上传数据列表</h3>
          <table class="upload-table">
            <thead>
              <tr>
                <th>数据源名称</th>
                <th>模态类型</th>
                <th>上传时间</th>
                <th>状态</th>
                <th>操作</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(item, index) in uploadedFiles" :key="index">
                <td>{{ item.name }}</td>
                <td>{{ item.type }}</td>
                <td>{{ item.uploadTime }}</td>
                <td>{{ item.status }}</td>
                <td>
                  <div class="btn-group">
                    <button class="btn-view" @click="handleView(item)">查看</button>
                    <button class="btn-delete" @click="handleDelete(item)">删除</button>
                    <button class="btn-reupload" @click="handleReupload(item)">重新上传</button>
                    <button class="btn-align" @click="handleAlign(item)">数据对齐</button>
                  </div>
                </td>
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

// 表单数据
const formData = ref({
  dataSourceName: '',
  modalType: 'text',
  orgName: '',
  file: null
});

// 上传进度
const uploadProgress = ref(0);

// 已上传文件列表
const uploadedFiles = ref([
  {
    name: '广医医院2023-Q4文本',
    type: '文本',
    uploadTime: '2025-08-10 15:30',
    status: '成功'
  }
]);

// 文件输入引用
const fileInput = ref(null);

// 计算属性：是否可以上传
const canUpload = computed(() => {
  return formData.value.dataSourceName && 
         formData.value.orgName && 
         formData.value.file;
});

// 触发文件选择
const triggerFileInput = () => {
  fileInput.value.click();
};

// 处理文件选择
const handleFileChange = (event) => {
  const file = event.target.files[0];
  if (file) {
    formData.value.file = file;
  }
};

// 处理文件拖放
const handleDrop = (event) => {
  const file = event.dataTransfer.files[0];
  if (file) {
    formData.value.file = file;
  }
};

// 处理上传
const handleUpload = async () => {
  if (!canUpload.value) return;
  
  // 模拟上传进度
  for (let i = 0; i <= 100; i += 10) {
    await new Promise(resolve => setTimeout(resolve, 200));
    uploadProgress.value = i;
  }
  
  // 添加上传记录
  uploadedFiles.value.unshift({
    name: formData.value.dataSourceName,
    type: formData.value.modalType,
    uploadTime: new Date().toLocaleString(),
    status: '成功'
  });
  
  // 重置表单
  resetForm();
};

// 处理取消
const handleCancel = () => {
  resetForm();
};

// 重置表单
const resetForm = () => {
  formData.value = {
    dataSourceName: '',
    modalType: 'text',
    orgName: '',
    file: null
  };
  uploadProgress.value = 0;
};

// 处理查看
const handleView = (item) => {
  console.log('查看文件:', item);
};

// 处理删除
const handleDelete = (item) => {
  const index = uploadedFiles.value.indexOf(item);
  if (index > -1) {
    uploadedFiles.value.splice(index, 1);
  }
};

// 处理重新上传
const handleReupload = (item) => {
  console.log('重新上传:', item);
};

// 处理数据对齐
const handleAlign = (item) => {
  console.log('数据对齐:', item);
};
</script>

<style lang="scss" scoped>
.data-upload {
  .container {
    margin: 1rem 2rem;
  }

  .upload-header {
    background-color: #fff;
    border-radius: 6px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    padding: 1rem;
    margin-bottom: 1rem;

    h2 {
      font-size: 1rem;
      margin-bottom: 0.5rem;
      color: #007bff;
    }

    p {
      font-size: 0.9rem;
      line-height: 1.4;
      color: #666;
    }
  }

  .upload-container {
    display: flex;
    gap: 1rem;
    min-height: 400px;
  }

  .left-panel, .right-panel {
    flex: 1;
    background-color: #fff;
    border-radius: 6px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    padding: 1rem;

    h3 {
      font-size: 1rem;
      margin-bottom: 0.5rem;
      color: #007bff;
    }
  }

  .drag-area {
    margin: 1rem 0;
    padding: 2rem;
    border: 2px dashed #aaa;
    border-radius: 6px;
    text-align: center;
    color: #777;
    cursor: pointer;

    &:hover {
      background-color: #f0f3f5;
    }
  }

  .info-field {
    margin-bottom: 0.8rem;

    label {
      display: block;
      margin-bottom: 0.3rem;
      font-weight: bold;
      color: #444;
    }

    input, select {
      width: 100%;
      padding: 0.4rem;
      border: 1px solid #ccc;
      border-radius: 4px;
    }
  }

  .upload-actions {
    display: flex;
    gap: 0.6rem;
    margin-top: 1rem;

    button {
      flex: 1;
      padding: 0.6rem;
      background-color: #28a745;
      color: #fff;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      font-size: 0.95rem;

      &:hover {
        opacity: 0.85;
      }

      &:disabled {
        opacity: 0.5;
        cursor: not-allowed;
      }
    }
  }

  .progress-area {
    margin-top: 1rem;
    font-size: 0.9rem;
    color: #666;
  }

  .upload-table {
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

    .btn-group {
      display: flex;
      gap: 0.3rem;

      button {
        padding: 0.3rem 0.6rem;
        border-radius: 4px;
        border: none;
        cursor: pointer;
        font-size: 0.85rem;

        &:hover {
          opacity: 0.85;
        }
      }

      .btn-view {
        background-color: #17a2b8;
        color: #fff;
      }

      .btn-delete {
        background-color: #dc3545;
        color: #fff;
      }

      .btn-reupload {
        background-color: #ffc107;
        color: #fff;
      }

      .btn-align {
        background-color: #007bff;
        color: #fff;
      }
    }
  }
}
</style>
