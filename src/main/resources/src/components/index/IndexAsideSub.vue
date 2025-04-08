<template>
  <el-submenu v-if="menu.list && menu.list.length >= 1" :index="menu.menuId + ''">
    <template slot="title">
      <span>{{ menu.name }}</span>
    </template>
    <sub-menu
      v-for="item in menu.list"
      :key="item.menuId"
      :menu="item"
      :dynamicMenuRoutes="dynamicMenuRoutes"
    ></sub-menu>
  </el-submenu>
  <el-menu-item v-else :index="menu.menuId + ''" @click="gotoRouteHandle(menu)">
    <span>{{ menu.name }}</span>
  </el-menu-item>
</template>

<script>
import SubMenu from "./IndexAsideSub.vue";
export default {
  name: "sub-menu",
  props: {
    menu: {
      type: Object,
      required: true
    },
    dynamicMenuRoutes: {
      type: Array,
      required: true
    }
  },
  components: {
    SubMenu
  },
  methods: {
    // 通过menuId与动态(菜单)路由进行匹配跳转至指定路由
    gotoRouteHandle(menu) {
      var route = this.dynamicMenuRoutes.filter(
        item => item.meta.menuId === menu.menuId
      );
      if (route.length >= 1) {
        if (route[0].component != null) {
          this.$router.replace({ name: route[0].name });
        } else {
          this.$router.push({ name: "404" });
        }
      }
    }
  }
};
</script>

<style lang="scss" scoped>
/* 基本样式，适用于所有屏幕尺寸 */
.el-menu {
  .el-submenu > .el-menu-item,
  .el-menu-item {
    padding: 12px 20px; /* 增加内边距，方便点击 */
    border-bottom: 1px solid #f0f0f0; /* 每个菜单项下方增加边框，提升分隔度 */
    &:last-child {
      border-bottom: none; /* 最后一个菜单项不需要边框 */
    }
  }

  .el-submenu .el-menu {
    .el-menu-item {
      padding-left: 40px; /* 子菜单项增加左侧内边距，表现出层级关系 */
    }
  }

  .el-submenu > .el-submenu__title {
    display: flex; /* 使用Flex布局 */
    align-items: center; /* 垂直居中 */
    span {
      flex-grow: 1; /* 让标题占满剩余空间 */
    }
  }
}

/* 针对移动端的样式调整 */
@media (max-width: 768px) {
  .el-menu {
    .el-submenu > .el-menu-item,
    .el-menu-item {
      font-size: 14px; /* 减小字体大小，适应小屏幕 */
    }

    .el-submenu .el-menu {
      .el-menu-item {
        padding-left: 30px; /* 子菜单项减少左侧内边距 */
      }
    }

    .el-submenu > .el-submenu__title {
      font-size: 14px; /* 调整子菜单标题的字体大小 */
      padding: 10px 20px; /* 调整子菜单标题的内边距 */
    }
  }
}
</style>

<style lang="scss" scoped>
/* 基本样式，适用于所有屏幕尺寸 */
.el-menu {
  .el-submenu > .el-menu-item,
  .el-menu-item {
    padding: 12px 20px; /* 增加内边距，方便点击 */
    border-bottom: 1px solid #f0f0f0; /* 每个菜单项下方增加边框，提升分隔度 */
    &:last-child {
      border-bottom: none; /* 最后一个菜单项不需要边框 */
    }
  }

  .el-submenu .el-menu {
    .el-menu-item {
      padding-left: 40px; /* 子菜单项增加左侧内边距，表现出层级关系 */
    }
  }

  .el-submenu > .el-submenu__title {
    display: flex; /* 使用Flex布局 */
    align-items: center; /* 垂直居中 */
    span {
      flex-grow: 1; /* 让标题占满剩余空间 */
    }
  }
}

/* 针对移动端的样式调整 */
@media (max-width: 768px) {
  .el-menu {
    .el-submenu > .el-menu-item,
    .el-menu-item {
      font-size: 14px; /* 减小字体大小，适应小屏幕 */
    }

    .el-submenu .el-menu {
      .el-menu-item {
        padding-left: 30px; /* 子菜单项减少左侧内边距 */
      }
    }

    .el-submenu > .el-submenu__title {
      font-size: 14px; /* 调整子菜单标题的字体大小 */
      padding: 10px 20px; /* 调整子菜单标题的内边距 */
    }
  }
}
</style>
