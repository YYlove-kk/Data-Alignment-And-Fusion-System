<template>
  <div class="homepage router-view-container">
    <div class="container">
      <MainContent />
      <Sidebar />
    </div>
  </div>
</template>

<script setup>
import MainContent from '../components/home/MainContent.vue'
import Sidebar from '../components/home/Sidebar.vue'
</script>

<style lang="scss" scoped>
.homepage {
  display: flex;
  flex-direction: column;
}

.container {
  display: flex;
  margin: 1rem 2rem;
  flex: 1;
}
</style>
