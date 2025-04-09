<template>
  <div class="main-content">
    <h2>主要知识图谱展示</h2>
    <div class="graph-visual">
      <p>这里将渲染庞大的医养康跨模态知识图谱</p>

      <div class="legend-box">
        <h4>关系与模态图例</h4>

        <div class="legend-item">
          <span class="color-swatch" style="background-color:red;"></span>
          患者-疾病(红色线)
        </div>
        <div class="legend-item">
          <span class="color-swatch" style="background-color:orange;"></span>
          疾病-影像(橙色线)
        </div>
        <div class="legend-item">
          <span class="color-swatch" style="background-color:purple;"></span>
          患者-患者(同一对象)(紫色线)
        </div>
        <div class="legend-item">
          <span class="color-swatch" style="background-color:blue;"></span>
          康复关联(蓝色线)
        </div>
        <div class="legend-item">
          <span class="color-swatch" style="background-color:gray;"></span>
          医疗机构归属(灰色线)
        </div>

        <div class="legend-item">
          <span class="shape-swatch diamond"></span>
          患者数据节点(菱形)
        </div>
        <div class="legend-item">
          <span class="shape-swatch square"></span>
          表单数据节点(正方形)
        </div>
        <div class="legend-item">
          <span class="triangle"></span>
          图像数据节点(三角形)
        </div>
        <div class="legend-item">
          <span class="trapezoid"></span>
          机构数据节点(梯形)
        </div>
        <div class="legend-item">
          <span class="shape-swatch circle"></span>
          属性数据节点(圆形)
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
// 組件邏輯
</script>

<style lang="scss" scoped>
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
}

.graph-visual {
  border: 1px dashed #ccc;
  height: 95%;
  margin-bottom: 1rem;
  display: flex;
  justify-content: center;
  align-items: center;
  color: #999;
  border-radius: 4px;
  position: relative;
}

.legend-box {
  position: absolute;
  bottom: 10px;
  right: 10px;
  background-color: #fafafa;
  border: 1px solid #ddd;
  border-radius: 4px;
  padding: 0.5rem 1rem;
  font-size: 0.85rem;
  line-height: 1.4;
  color: #333;
  box-shadow: 0 1px 2px rgba(0,0,0,0.1);

  h4 {
    margin-bottom: 0.3rem;
    color: #007bff;
  }
}

.legend-item {
  display: flex;
  align-items: center;
  margin: 0.3rem 0;
}

.color-swatch {
  width: 12px;
  height: 12px;
  border-radius: 2px;
  display: inline-block;
  margin-right: 0.4rem;
}

.shape-swatch {
  width: 16px;
  height: 16px;
  display: inline-block;
  margin-right: 0.4rem;
  border: 2px solid #000;
  margin-top: 2px;

  &.diamond {
    transform: rotate(45deg);
  }

  &.circle {
    border-radius: 50%;
  }
}

.triangle {
  width: 0;
  height: 0;
  border-left: 8px solid transparent;
  border-right: 8px solid transparent;
  border-bottom: 14px solid #000;
  margin-right: 0.4rem;
  margin-top: 1px;
}

.trapezoid {
  width: 0;
  height: 0;
  border-bottom: 16px solid #000;
  border-left: 6px solid transparent;
  border-right: 6px solid transparent;
  margin-right: 0.4rem;
  margin-top: 1px;
}
</style> 