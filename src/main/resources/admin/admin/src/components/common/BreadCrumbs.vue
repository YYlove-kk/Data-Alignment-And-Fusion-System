<template>
  <el-breadcrumb class="app-breadcrumb" separator="✌">
    <transition-group name="breadcrumb" class="box" :style="1==1?'justifyContent:flex-start;':1==2?'justifyContent:center;':'justifyContent:flex-end;'">
      <el-breadcrumb-item v-for="(item,index) in levelList" :key="item.path">
        <span v-if="item.redirect==='noRedirect'||index==levelList.length-1" class="no-redirect">{{ item.name }}</span>
        <a v-else @click.prevent="handleLink(item)">{{ item.name }}</a>
      </el-breadcrumb-item>
    </transition-group>
  </el-breadcrumb>
</template>

<script>
import pathToRegexp from 'path-to-regexp'
import { generateTitle } from '@/utils/i18n'
export default {
  data() {
    return {
      levelList: null
    }
  },
  watch: {
    $route() {
      this.getBreadcrumb()
    }
  },
  created() {
    this.getBreadcrumb()
    this.breadcrumbStyleChange()
  },
  methods: {
    generateTitle,
    getBreadcrumb() {
      // only show routes with meta.title
      let route = this.$route
      let matched = route.matched.filter(item => item.meta)
      const first = matched[0]
      matched = [{ path: '/index' }].concat(matched)

      this.levelList = matched.filter(item => item.meta)
    },
    isDashboard(route) {
      const name = route && route.name
      if (!name) {
        return false
      }
      return name.trim().toLocaleLowerCase() === 'Index'.toLocaleLowerCase()
    },
    pathCompile(path) {
      // To solve this problem https://github.com/PanJiaChen/vue-element-admin/issues/561
      const { params } = this.$route
      var toPath = pathToRegexp.compile(path)
      return toPath(params)
    },
    handleLink(item) {
      const { redirect, path } = item
      if (redirect) {
        this.$router.push(redirect)
        return
      }
      this.$router.push(path)
    },
    breadcrumbStyleChange(val) {
      this.$nextTick(()=>{
        document.querySelectorAll('.app-breadcrumb .el-breadcrumb__separator').forEach(el=>{
          el.innerText = "✌"
          el.style.color = "#C0C4CC"
        })
        document.querySelectorAll('.app-breadcrumb .el-breadcrumb__inner a').forEach(el=>{
          el.style.color = "rgba(117, 113, 249, 1)"
        })
        document.querySelectorAll('.app-breadcrumb .el-breadcrumb__inner .no-redirect').forEach(el=>{
          el.style.color = "rgba(179, 179, 179, 1)"
        })

        let str = "vertical"
        if("vertical" === str) {
          let headHeight = "40px"
          headHeight = parseInt(headHeight) + 10 + 'px'
          document.querySelectorAll('.app-breadcrumb').forEach(el=>{
            el.style.marginTop = headHeight
          })
        }

      })
    },
  }
}
</script>

<style lang="scss" scoped>
.app-breadcrumb {
  display: block;
  font-size: 14px;
  line-height: 50px;
  height:40px;
  //backgroundColor:${template.back.breadcrumb.boxBackgroundColor};
  borderRadius:0px;
  padding:0px 0px 0px 450px;
  boxShadow:0px 0px 0px #f903d4;
  borderWidth:2px;
  borderStyle:none none dashed none ;
  borderColor:rgba(117, 113, 249, 1);

  .box {
    display: flex;
    width: 100%;
    height: 100%;
    justify-content: flex-start;
    align-items: center;
  }

  .no-redirect {
    color: #97a8be;
    cursor: text;
  }
}

/* 移动端适配 */
@media (max-width: 768px) {
  .app-breadcrumb {
    height: 40px; // 确保高度一致
    background-color: #f3f3f3; // 假定背景色与示例中相同
    box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
    border-radius: 0px; // 如有必要，调整边框圆角
    display: flex; // 使用弹性盒子布局
    align-items: center; // 垂直居中对齐所有子元素
    padding: 0 15px; // 减少左内边距，保证内容在小屏幕上也能左对齐
    justify-content: flex-start; // 确保面包屑项在容器中左对齐
    margin: 0; // 重置外边距
    box-sizing: border-box; // 确保padding不会影响总宽度
    border-bottom: 2px dashed rgba(117, 113, 249, 1); // 根据示例图添加边框样式
  }

  .app-breadcrumb .el-breadcrumb__item {
    margin: 0; // 重置默认的外边距
    white-space: nowrap; // 防止文字换行
    overflow: hidden; // 隐藏溢出文本
    text-overflow: ellipsis; // 显示省略号
  }

  .app-breadcrumb .el-breadcrumb__inner a,
  .app-breadcrumb .el-breadcrumb__inner .no-redirect {
    font-size: 14px; // 调整字体大小以适应移动端阅读
  }

  .app-breadcrumb .el-breadcrumb__separator {
    font-size: 14px; // 保持分隔符的字体大小一致
    color: #C0C4CC; // 分隔符颜色
  }
}

</style>
