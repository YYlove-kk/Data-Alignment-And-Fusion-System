<template>
  <div>
    <!-- 上传文件组件 -->
    <el-upload
      ref="upload"
      :action="getActionUrl"
      list-type="picture-card"
      :multiple="multiple"
      :limit="limit"
      :headers="myHeaders"
      :file-list="fileList"
      :on-exceed="handleExceed"
      :on-preview="handleUploadPreview"
      :on-remove="handleRemove"
      :on-success="handleUploadSuccess"
      :on-error="handleUploadErr"
      :before-upload="handleBeforeUpload"
    >
      <i class="el-icon-plus"></i>
      <div slot="tip" class="el-upload__tip" style="color:#838fa1;">{{tip}}</div>
    </el-upload>
    <el-dialog :visible.sync="dialogVisible" size="tiny" append-to-body>
      <img width="100%" :src="dialogImageUrl" alt>
    </el-dialog>
  </div>
</template>
<script>
import storage from "@/utils/storage";
import base from "@/utils/base";
export default {
  data() {
    return {
      // 查看大图
      dialogVisible: false,
      // 查看大图
      dialogImageUrl: "",
      // 组件渲染图片的数组字段，有特殊格式要求
      fileList: [],
      fileUrlList: [],
      myHeaders:{}
    };
  },
  props: ["tip", "action", "limit", "multiple", "fileUrls"],
  mounted() {
    this.init();
    this.myHeaders= {
      'Token':storage.get("Token")
    }
  },
  watch: {
    fileUrls: function(val, oldVal) {
      //   console.log("new: %s, old: %s", val, oldVal);
      this.init();
    }
  },
  computed: {
    // 计算属性的 getter
    getActionUrl: function() {
      // return base.url + this.action + "?token=" + storage.get("token");
      return `/${this.$base.name}/` + this.action;
    }
  },
  methods: {
    // 初始化
    init() {
      //   console.log(this.fileUrls);
      if (this.fileUrls) {
        this.fileUrlList = this.fileUrls.split(",");
        let fileArray = [];
        this.fileUrlList.forEach(function(item, index) {
          var url = item;
          var name = index;
          var file = {
            name: name,
            url: url
          };
          fileArray.push(file);
        });
        this.setFileList(fileArray);
      }
    },
    handleBeforeUpload(file) {
	
    },
    // 上传文件成功后执行
    handleUploadSuccess(res, file, fileList) {
      if (res && res.code === 0) {
        fileList[fileList.length - 1]["url"] =
          this.$base.url + "upload/" + file.response.file;
        this.setFileList(fileList);
        this.$emit("change", this.fileUrlList.join(","));
      } else {
        this.$message.error(res.msg);
      }
    },
    // 图片上传失败
    handleUploadErr(err, file, fileList) {
      this.$message.error("文件上传失败");
    },
    // 移除图片
    handleRemove(file, fileList) {
      this.setFileList(fileList);
      this.$emit("change", this.fileUrlList.join(","));
    },
    // 查看大图
    handleUploadPreview(file) {
      this.dialogImageUrl = file.url;
      this.dialogVisible = true;
    },
    // 限制图片数量
    handleExceed(files, fileList) {
      this.$message.warning(`最多上传${this.limit}张图片`);
    },
    // 重新对fileList进行赋值
    setFileList(fileList) {
      var fileArray = [];
      var fileUrlArray = [];
      // 有些图片不是公开的，所以需要携带token信息做权限校验
      var token = storage.get("token");
      fileList.forEach(function(item, index) {
        var url = item.url.split("?")[0];
        var name = item.name;
        var file = {
          name: name,
          url: url + "?token=" + token
        };
        fileArray.push(file);
        fileUrlArray.push(url);
      });
      this.fileList = fileArray;
      this.fileUrlList = fileUrlArray;
    }
  }
};
</script>
<style lang="scss" scoped>
/* 先定义上传组件的基本样式 */
.el-upload {
  border: 1px dashed #d9d9d9;
  border-radius: 6px;
  cursor: pointer;
  position: relative;
  overflow: hidden;
}

.el-upload__tip {
  color: #838fa1;
  font-size: 14px;
  line-height: 22px;
  margin-top: 10px;
  text-align: center;
}

/* 定义媒体查询，专门针对移动端设备 */
@media (max-width: 768px) {
  .el-upload {
    border-width: 0.5px; /* 在移动端减小边框宽度 */
    border-radius: 4px; /* 在移动端减小边角半径 */
  }

  .el-upload__tip {
    font-size: 12px; /* 在移动端减小提示文本的字体大小 */
    padding: 0 10px; /* 添加一些内边距以防止文本溢出 */
  }

  .el-upload-list__item {
    width: 48%; /* 调整移动端图片列表项的宽度，以便在一行内显示更多项 */
    margin-right: 2%; /* 添加列表项之间的间距 */
    margin-bottom: 10px; /* 增加列表项底部的间距 */
  }

  .el-upload-list {
    display: flex; /* 使用Flex布局 */
    flex-wrap: wrap; /* 允许列表项换行 */
    justify-content: space-between; /* 分散对齐列表项 */
  }

  .el-dialog__wrapper {
    padding: 0 10px; /* 对话框外边距，避免触碰屏幕边缘 */
  }

  .el-dialog {
    width: auto; /* 对话框宽度自适应 */
    max-width: 90%; /* 最大宽度，确保对话框不会过大 */
  }

  .el-dialog img {
    width: 100%; /* 图片宽度自适应，确保图片在对话框内完整显示 */
  }
}

</style>
