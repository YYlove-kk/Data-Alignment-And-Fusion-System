<template>
  <header>
    <div class="logo-title">
      <span class="system-title">多源医养康跨模态数据对齐融合系统</span>
    </div>

    <div class="search-bar">
      <input type="text" placeholder="搜索(患者ID/疾病名称/机构名称)" />
    </div>

    <div class="user-info">
      <nav>
        <ul>
          <li v-for="(item, index) in tabType" :key="index">
            <a
              :class="index === activeType ? 'active' : ''"
              href="#"
              @click="menuSelect(index)"
              >{{ item.title }}</a
            >
          </li>
        </ul>
      </nav>
    </div>
  </header>
</template>

<script setup>
import { ref, watch } from "vue";
import { useRouter, useRoute } from "vue-router";

// 組件邏輯
const activeType = ref(0);
const router = useRouter();
const route = useRoute();
watch(
  () => route.name,
  () => {
    console.log("當前路由名稱:", route.name, "當前路由路徑:", route);
    if (!route.name) {
      activeType.value = 0;
    } else {
      activeType.value = tabType.findIndex((item) => item.name === route.name);
    }
  }
);

const tabType = [
  { title: "首页", src: "/", name: "home" },
  { title: "数据上传", src: "dataUpload", name: "dataUpload" },
  { title: "数据对齐", src: "/dataAlignment", name: "dataAlignment" },
  {
    title: "多模态知识图谱构建与融合",
    src: "/multiModalKnowledgeGraph",
    name: "multiModalKnowledgeGraph",
  },
  {
    title: "患者数据分析",
    src: "/patientDataAnalysis",
    name: "patientDataAnalysis",
  },
  { title: "系统设置", src: "/systemSettings", name: "systemSettings" },
];
const menuSelect = (type) => {
  activeType.value = type;
  router.push(tabType[type].src);
  console.log("選中的選單索引11111:", type);
  console.log("選中的路由名稱:", tabType[type].name);
};
</script>

<style lang="scss" scoped>
header {
  background-color: #ffffff;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  padding: 1rem 2rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.logo-title {
  display: flex;
  align-items: center;

  img {
    height: 40px;
    margin-right: 10px;
  }
}

.system-title {
  font-size: 1.2rem;
  font-weight: bold;
  color: #007bff;
}

.search-bar {
  flex: 1;
  margin: 0 1rem;

  input {
    width: 80%;
    padding: 0.4rem 0.6rem;
    border: 1px solid #ccc;
    border-radius: 4px;
  }
}

.user-info {
  font-size: 0.9rem;
  color: #555;
}

nav {
  ul {
    display: flex;
    list-style: none;
    gap: 1rem;
  }

  li a {
    text-decoration: none;
    color: #333;
    padding: 0.3rem 0.6rem;
    border-radius: 4px;

    &:hover,
    &.active {
      background-color: #007bff;
      color: #fff;
    }
  }
}
</style>
