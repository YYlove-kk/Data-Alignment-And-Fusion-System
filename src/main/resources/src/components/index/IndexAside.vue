<template>
  <el-aside class="index-aside" width="200px">
    <div class="index-aside-inner">
      <el-menu default-active="1">
        <el-menu-item @click="backDiagnose()" index="1">
          <!-- <i class="el-icon-s-home"></i> -->
          首页
        </el-menu-item>
        <sub-menu
          v-for="menu in menuList"
          :key="menu.menuId"
          :menu="menu"
          :dynamicMenuRoutes="dynamicMenuRoutes"
        ></sub-menu>
      </el-menu>
    </div>
  </el-aside>
</template>
<script>
import SubMenu from "@/components/index/IndexAsideSub";
export default {
  data() {
    return {
      menuList: [],
      dynamicMenuRoutes: []
    };
  },
  components: {
    SubMenu
  },
  mounted() {
    // 获取动态菜单数据并且渲染
    this.menuList = JSON.parse(sessionStorage.getItem("menuList") || "[]");
    this.dynamicMenuRoutes = JSON.parse(
      sessionStorage.getItem("dynamicMenuRoutes") || "[]"
    );
  },
  methods: {
    menuHandler() {
      this.$router.push({path:'/diagnose'})
    },
  }
};
</script>
<style lang="scss" scoped>
.index-aside {
  margin-top: 0; /* 移动端不需要额外的上边距 */
  .index-aside-inner {
    width: 100%; /* 移动端全宽度展示 */
    height: 100vh; /* 设置高度为视窗高度，保证足够空间显示菜单项 */
    overflow-y: auto; /* 允许在Y轴方向上滚动 */
    position: fixed; /* 固定定位，确保用户滚动页面时侧边栏仍可见 */
    top: 0; /* 顶部对齐 */
    left: 0; /* 左侧对齐 */
    z-index: 1000; /* 高于页面其他元素 */
    background-color: #FFFFFF; /* 背景颜色，提高可读性 */
    box-shadow: 0 2px 4px rgba(0,0,0,0.2); /* 添加阴影以增加层次感 */
    padding-top: 60px; /* 为顶部导航栏留出空间 */
    transition: transform 0.3s ease; /* 平滑的展示和隐藏动画 */

    /* 移动端导航栏默认隐藏，通过添加一个类来控制显示 */
    transform: translateX(-100%);
    &.is-active {
      transform: translateX(0);
    }

    el-menu {
      border-right: 1px solid #ebeef5; /* 给菜单添加右侧边框，区分菜单项和内容 */
      .el-menu-item, .el-submenu__title {
        padding: 10px 20px; /* 减少内边距 */
        font-size: 14px; /* 适应移动端的字体大小 */
      }

      /* 子菜单样式调整 */
      .el-submenu {
        .el-submenu__title {
          .el-icon-arrow-down {
            transition: transform 0.3s ease; /* 子菜单展开图标的平滑过渡 */
          }
          &.is-opened .el-icon-arrow-down {
            transform: rotate(180deg); /* 点击展开旋转图标 */
          }
        }
      }
    }
  }
}

/* 控制侧边栏的显示隐藏 */
.toggle-aside {
  position: fixed;
  top: 20px;
  left: 20px;
  z-index: 1001; /* 确保在侧边栏上面 */
  cursor: pointer;
}

@media (min-width: 769px) {
  /* 大屏幕时恢复原有样式 */
  .index-aside {
    .index-aside-inner {
      position: static;
      width: 200px; /* 原始宽度 */
      height: auto; /* 自动高度 */
      padding-top: 0;
      overflow-y: visible; /* 不需要滚动 */
      transform: none;
      box-shadow: none;
    }

    .toggle-aside {
      display: none; /* 大屏幕时不显示控制按钮 */
    }
  }
}
</style>


