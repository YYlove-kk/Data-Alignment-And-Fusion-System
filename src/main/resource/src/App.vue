<template>
  <div class="app">
    <Header />
    <RouterView />
    <Footer />
  </div>
</template>

<script setup>
import { RouterView } from 'vue-router'
import Header from './components/Header.vue'
import Footer from './components/Footer.vue'
</script>

<style lang="scss">
.app {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

/* 確保主內容區域能夠彈性佔據剩餘空間 */
.router-view-container {
  flex: 1;
  display: flex;
  flex-direction: column;
}
</style>
