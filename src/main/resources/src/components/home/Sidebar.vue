<template>
  <div class="sidebar-right">
    <h2>节点/关系详情</h2>
    <p>当用户点击知识图谱中的某节点或连线时，在这里显示更详细的属性与关联信息。</p>
    <p>例如：  
      - 节点类型(患者/疾病/影像) <br/>
      - 节点名称、描述、相关统计 <br/>
      - 相似度/置信度 <br/>
    </p>
  </div>
</template>

<script setup>
// 組件邏輯
</script>

<style lang="scss" scoped>
.sidebar-right {
  flex: 0 0 280px;
  background-color: #fff;
  border-radius: 6px;
  margin-left: 1rem;
  padding: 1rem;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1);

  h2 {
    font-size: 1rem;
    margin-bottom: 1rem;
    color: #007bff;
  }

  p {
    font-size: 0.9rem;
    line-height: 1.4;
    color: #666;
  }
}
</style> 