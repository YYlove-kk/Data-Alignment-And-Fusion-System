<template>
  <div class="loginIn">

      <div v-if="isInit" class="features-container">
          <div class="featuresLeft">
              <el-button class="feature-button" type="primary" @click="handleUpgradeClick">任务升级</el-button>
          </div>
          <div class="featuresRight">
              <el-button class="feature-button" type="primary"  @click="handleDiagnosisClick">数据诊断</el-button>
          </div>
      </div>



    <div v-if="isShowUpdateLogin" :class="2 == 1 ? 'left' : 2 == 2 ? 'left center' : 'left right'" style="background-color: rgba(0, 0, 0, 0.75);">

      <el-form class="login-form" label-position="left" :label-width="1 == 3 ? '56px' : '0px'">

        <div class="logo-container" style="text-align: center; height: 100px;margin-bottom: 20px; ">
          <img src="../assets/img/logohead.png" style="width: 120px; height: 140px;">
        </div>

        <div class="title-container"><h3 class="title" style="color: rgba(255, 255, 255, 0.87);margin-bottom: 20px; ">电池任务升级系统</h3></div>
        <el-form-item :label="1 == 3 ? '用户名' : ''" :class="'style'+1">
          <span v-if="1 != 3" class="svg-container" style="color:rgba(255, 255, 255, 0.87);line-height:44px"><svg-icon icon-class="user" /></span>
          <el-input placeholder="请输入用户名" name="username" type="text" v-model="rulesForm.username" style="background-color: rgba(255, 255, 255, 0.12); color: rgba(255, 255, 255, 0.87)" />
        </el-form-item>
        <el-form-item :label="1 == 3 ? '密码' : ''" :class="'style'+1">
          <span v-if="1 != 3" class="svg-container" style="color:rgba(255, 255, 255, 0.87);line-height:44px"><svg-icon icon-class="password" /></span>
          <el-input placeholder="请输入密码" name="password" type="password" v-model="rulesForm.password" style="background-color: rgba(255, 255, 255, 0.12); color: rgba(255, 255, 255, 0.87)" />
        </el-form-item>
        <el-form-item label="角色" prop="loginInRole" class="role">
          <el-radio v-for="item in menus" v-bind:key="item.roleName" v-model="rulesForm.role" :label="item.roleName" style="color: rgba(255, 255, 255, 0.87)">{{item.roleName}}</el-radio>
        </el-form-item>
        <div>
        <el-button type="primary" @click="adminLogin()" style="padding:0; font-size:16px; border-radius:4px; height:44px; line-height:44px; width:100%; background-color:#455A64; border-color:#455A64; color:rgba(255, 255, 255, 0.87)">登录</el-button>
        </div>
        <div>
          <el-button type="primary" @click="back()" style="padding:0; margin-top: 10px; font-size:16px; border-radius:4px; height:44px; line-height:44px; width:100%; background-color:#455A64; border-color:#455A64; color:rgba(255, 255, 255, 0.87)">返回</el-button>
        </div>
          <el-form-item class="setting">
          <div style="color:rgba(255, 255, 255, 0.87)" class="register" @click="register('yonghu')">注册用户</div>
        </el-form-item>

      </el-form>
    </div>

    <div v-if="isShowDiagnosisLogin" :class="2 == 1 ? 'left' : 2 == 2 ? 'left center' : 'left right'" style="background-color: rgba(0, 0, 0, 0.75);">
      <el-form class="login-form" label-position="left" :label-width="1 == 3 ? '56px' : '0px'">
        <div class="logo-container" style="text-align: center; height: 100px;margin-bottom: 20px; ">
          <img src="../assets/img/logohead.png" style="width: 120px; height: 140px;">
        </div>

        <div class="title-container"><h3 class="title" style="color: rgba(255, 255, 255, 0.87);margin-bottom: 20px; ">电池状态管理系统</h3></div>
        <el-form-item :label="1 == 3 ? '用户名' : ''" :class="'style'+1">
          <span v-if="1 != 3" class="svg-container" style="color:rgba(255, 255, 255, 0.87); line-height:44px"><svg-icon icon-class="user" /></span>
          <el-input placeholder="请输入用户名" name="username" type="text" v-model="rulesForm.username" style="background-color: rgba(255, 255, 255, 0.12); color: rgba(255, 255, 255, 0.87)" />
        </el-form-item>
        <el-form-item :label="1 == 3 ? '密码' : ''" :class="'style'+1">
          <span v-if="1 != 3" class="svg-container" style="color:rgba(255, 255, 255, 0.87); line-height:44px"><svg-icon icon-class="password" /></span>
          <el-input placeholder="请输入密码" name="password" type="password" v-model="rulesForm.password" style="background-color: rgba(255, 255, 255, 0.12); color: rgba(255, 255, 255, 0.87)" />
        </el-form-item>
        <div>
          <el-button type="primary" @click="userLogin()" style="padding:0; font-size:16px; border-radius:4px; height:44px; line-height:44px; width:100%; background-color:#455A64; border-color:#455A64; color:rgba(255, 255, 255, 0.87)">登录</el-button>
        </div>
        <div>
          <el-button type="primary" @click="back()" style="padding:0;margin-top: 10px; font-size:16px; border-radius:4px; height:44px; line-height:44px; width:100%; background-color:#455A64; border-color:#455A64; color:rgba(255, 255, 255, 0.87)">返回</el-button>
        </div>
      </el-form>
    </div>


  </div>
</template>

<script>
import menu from "@/utils/menu";
export default {
  data() {
    return {
      rulesForm: {
        username: "",
        password: "",
        role: "",
        code: '',
      },
      isShowDiagnosisLogin:false,
      isShowUpdateLogin:false,
      isShowLogin:false,
      isInit:true,

      menus: [],
      tableName: "",
      codes: [{
        num: 1,
        color: '#000',
        rotate: '10deg',
        size: '16px'
      },{
        num: 2,
        color: '#000',
        rotate: '10deg',
        size: '16px'
      },{
        num: 3,
        color: '#000',
        rotate: '10deg',
        size: '16px'
      },{
        num: 4,
        color: '#000',
        rotate: '10deg',
        size: '16px'
      }],
    };
  },
  mounted() {
    let menus = menu.list();
    this.menus = menus;
  },
  created() {
    this.setInputColor()
    this.getRandCode()
  },
  methods: {
    setInputColor(){
      this.$nextTick(()=>{
        document.querySelectorAll('.loginIn .el-input__inner').forEach(el=>{
          el.style.backgroundColor = "rgba(255, 255, 255, 0.66)"
          el.style.color = "rgba(117, 113, 249, 1)"
          el.style.height = "44px"
          el.style.lineHeight = "44px"
          el.style.borderRadius = "4px"
        })
        document.querySelectorAll('.loginIn .style3 .el-form-item__label').forEach(el=>{
          el.style.height = "44px"
          el.style.lineHeight = "44px"
        })
        document.querySelectorAll('.loginIn .el-form-item__label').forEach(el=>{
          el.style.color = "rgba(18, 18, 18, 1)"
        })
        setTimeout(()=>{
          document.querySelectorAll('.loginIn .role .el-radio__label').forEach(el=>{
            el.style.color = "#fff"
          })
        },350)
      })

    },
    register(){
      // 1. 是否选择了“管理员”角色，如果没选择则弹窗提示
      if (this.rulesForm.role !== '管理员') {
        this.$message.error('只有管理员才可以注册用户');
        return;
      }

      // 2. 是否输入了用户名和密码，如果没有则弹窗提示
      if (!this.rulesForm.username || !this.rulesForm.password) {
        this.$message.error('请输入用户名和密码');
        return;
      }

      // 3. 进行管理员登录校验
      this.$http.post('admin/login', {
        username: this.rulesForm.username,
        password: this.rulesForm.password
      }).then(response => {
        const data = response.data;
        if (data && data.status === 0) {
          // 4. 登录成功，跳转到注册页面
          this.$message.success('管理员登录成功');
          this.$router.replace({ path: '/register' });
        } else {
          this.$message.error('信息校验失败，无法进行注册');
        }
      }).catch(error => {
        console.error("Login error:", error);
        this.$message.error("登录过程中发生错误");
      });
    },
    // 电池任务下发系统登录
    adminLogin() {
    //判断如果选择的是用户，则进入用户任务下发界面，如果是管理员，则进入管理员对下发任务的审核界面
      console.log("Task Login");
      console.log("Role: " + this.rulesForm.role);

      // 假设登录验证成功，并且从后端获取到了role和adminName
      let role = this.rulesForm.role; // 假设这是从表单中获取的角色
      let adminName = this.rulesForm.username; // 假设这是从表单中获取的用户名

      // 在这里添加登录逻辑
      // 假设登录成功后，执行以下代码来更新存储的role和adminName
      this.$storage.set('role', role);
      this.$storage.set('adminName', adminName);
      this.$project.projectName = "任务升级管理系统";

      // 根据角色跳转到不同的路由
      if (role === '管理员') {
        this.$http.post('admin/login', {
          username: this.rulesForm.username,
          password: this.rulesForm.password
        }).then(response => {
          const data = response.data;
          if (data && data.status === 0) {
            this.$storage.set("sessionTable", this.tableName);
            this.$storage.set("adminName", this.rulesForm.username);
            this.$router.replace({ path: '/adminTaskManage' });
          } else {
            this.$message.error(data.message);
          }
        }).catch(error => {
          console.error("Login error:", error);
          this.$message.error("An error occurred during the login process.");
        });
      } else if (role === '用户') {
        this.$http.post('users/login', {
          username: this.rulesForm.username,
          password: this.rulesForm.password
        }).then(response => {
          const data = response.data;
          if (data && data.status === 0) {
            this.$storage.set("sessionTable", this.tableName);
            this.$storage.set("adminName", this.rulesForm.username);
            this.$router.replace({ path: '/userTask' });
          } else {
            this.$message.error(data.message);
          }
        }).catch(error => {
          console.error("Login error:", error);
          this.$message.error("An error occurred during the login process.");
        });
      } else {
        // 如果没有选择或选择了其他选项，可以在这里处理
        this.$message.error('请选择有效的角色进行登录。');
      }

    },

    userLogin() {
      console.log("User Login");
      console.log("username: " + this.rulesForm.username);
      console.log("password: " + this.rulesForm.password);

      this.$http.post('users/login', {
        username: this.rulesForm.username,
        password: this.rulesForm.password
      }).then(response => {
        const data = response.data;
        if (data && data.status === 0) {
          console.log("Login ok1");
          this.$storage.set("role", "用户");
          this.$storage.set("sessionTable", this.tableName);
          this.$storage.set("adminName", this.rulesForm.username);
          this.$router.replace({ path: 'diagnose' });
          console.log("Login ok2");
        } else {
          this.$message.error(data.message);
        }
      }).catch(error => {
        console.error("Login error:", error);
        this.$message.error("An error occurred during the login process.");
      });
    },


    handleUpgradeClick() {
      console.log("任务升级按钮被点击");
      this.isInit = false;
      this.isShowUpdateLogin = true;
    },
    handleDiagnosisClick() {
      console.log("数据诊断按钮被点击");
      // 在这里添加数据诊断的逻辑
      this.isInit = false;
      this.isShowDiagnosisLogin = true;
    },
    back(){
      this.isInit = true;
      this.isShowDiagnosisLogin = false;
      this.isShowUpdateLogin = false;
      this.rulesForm.username="";
      this.rulesForm.password="";
    },

    getRandCode(len = 4){
      this.randomString(len)
    },
    randomString(len = 4) {
      let chars = [
          "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k",
          "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v",
          "w", "x", "y", "z", "A", "B", "C", "D", "E", "F", "G",
          "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R",
          "S", "T", "U", "V", "W", "X", "Y", "Z", "0", "1", "2",
          "3", "4", "5", "6", "7", "8", "9"
      ]
      let colors = ["0", "1", "2","3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e", "f"]
      let sizes = ['14', '15', '16', '17', '18']

      let output = [];
      for (let i = 0; i < len; i++) {
        // 随机验证码
        let key = Math.floor(Math.random()*chars.length)
        this.codes[i].num = chars[key]
        // 随机验证码颜色
        let code = '#'
        for (let j = 0; j < 6; j++) {
          let key = Math.floor(Math.random()*colors.length)
          code += colors[key]
        }
        this.codes[i].color = code
        // 随机验证码方向
        let rotate = Math.floor(Math.random()*60)
        let plus = Math.floor(Math.random()*2)
        if(plus == 1) rotate = '-'+rotate
        this.codes[i].rotate = 'rotate('+rotate+'deg)'
        // 随机验证码字体大小
        let size = Math.floor(Math.random()*sizes.length)
        this.codes[i].size = sizes[size]+'px'
      }
    },
  }
};
</script>

<style lang="scss" scoped>
.loginIn {
  /* 确保loginIn覆盖整个页面 */
  height: 100vh;
  display: flex;
  justify-content: center;
  align-items: center;
  background-image: url('../assets/img/loginBG.png');
  background-size: cover;
  background-position: center;
  backdrop-filter: blur(10px); /* 增大背景虚化效果 */

  .left {
    position: absolute;
    left: 0;
    top: 0;
    width: 360px;
    height: 100%;

    .login-form {
      background-color: rgba(66, 66, 66, 0.92);
      width: 100%;
      right: inherit;
      padding: 0 12px;
      box-sizing: border-box;
      display: flex;
      justify-content: center;
      flex-direction: column;
    }

    .logo-container {
      position: relative; // 使用绝对定位使logo固定在顶部中心
      top: 0; // 位于顶部
      left: 50%; // 水平居中
      transform: translateX(-50%); // 精确居中调整
      width: 120px; // logo的宽度
      height: auto; // 自动调整高度以保持图片比例
      margin-top: 0px; // 根据需要调整顶部间距

      img {
        width: 100%; // 图片宽度撑满容器
        height: auto; // 高度自适应
      }
    }

    .title-container {
      text-align: center;
      font-size: 24px;

      .title {
        margin: 0px 0;
      }
    }

    .el-form-item {
      position: relative;

      .svg-container {
        padding: 6px 5px 6px 15px;
        color: #889aa4;
        vertical-align: middle;
        display: inline-block;
        position: absolute;
        left: 0;
        top: 0;
        z-index: 1;
        padding: 0;
        line-height: 40px;
        width: 30px;
        text-align: center;
      }

      .el-input {
        display: inline-block;
        height: 40px;
        width: 100%;
        margin-bottom: 0px; /* 或者更小的值，根据实际效果调整 */

        & /deep/ input {
          background: transparent;
          border: 0px;
          -webkit-appearance: none;
          padding: 0 15px 0 30px;
          color: #fff;
          height: 40px;
        }
      }
    }
  }

  .center {
    position: absolute;
    left: 50%;
    top: 50%;
    width: 360px;
    transform: translate3d(-50%,-50%,0);
    height: 446px;
    border-radius: 8px;
  }

  .right {
    position: absolute;
    left: inherit;
    right: 0;
    top: 0;
    width: 360px;
    height: 100%;
  }

  .code {
    .el-form-item__content {
      position: relative;

      .getCodeBt {
        position: absolute;
        right: 0;
        top: 0;
        line-height: 40px;
        width: 100px;
        background-color: rgba(51,51,51,0.4);
        color: #fff;
        text-align: center;
        border-radius: 0 4px 4px 0;
        height: 40px;
        overflow: hidden;

        span {
          padding: 0 5px;
          display: inline-block;
          font-size: 16px;
          font-weight: 600;
        }
      }

      .el-input {
        & /deep/ input {
          padding: 0 130px 0 30px;
        }
      }
    }
  }

  .setting {
    & /deep/ .el-form-item__content {
      padding: 0 15px;
      box-sizing: border-box;
      line-height: 32px;
      height: 32px;
      font-size: 14px;
      color: #999;
      margin: 0 !important;

      .register, .reset {
        cursor: pointer;
        &:hover {
          color: #fff;
        }
      }

      .register {
        float: left;
        width: 50%;
        height: 20%;
      }

      .reset {
        float: right;
        width: 50%;
        text-align: right;
      }
    }
  }

  .style2 {
    padding-left: 30px;

    .svg-container {
      left: -30px !important;
    }

    .el-input {
      & /deep/ input {
        padding: 0 15px !important;
      }
    }
  }

  .code.style2, .code.style3 {
    .el-input {
      & /deep/ input {
        padding: 0 115px 0 15px;
      }
    }
  }

  .style3 {
    & /deep/ .el-form-item__label {
      padding-right: 6px;
    }

    .el-input {
      & /deep/ input {
        padding: 0 15px !important;
      }
    }
  }

  .role {
    & /deep/ .el-form-item__label {
      width: 56px !important;
    }

    & /deep/ .el-radio {
      margin-right: 12px;
    }

    /* 减少角色选择与下一个元素（登录按钮）之间的距离 */
    margin-bottom: 0px; /* 或者更小的值，根据实际效果调整 */
  }


  /* 新增装饰元素样式 */
  .features-container {
    display: flex;
    height: 100vh;
    width: 100vh;
    justify-content: center; /* Center the items horizontally */
    align-items: center; /* Center the items vertically */
    gap: 100px; /* 在按钮之间增加空隙 */
  }

  .features-container.featuresLeft {
    flex-basis: 50%; /* Each child takes up 50% of the container */
    padding-right: 20px; /* Adds some padding to the right side for spacing */
  }

  .features-container.featuresRight {
    flex-basis: 50%; /* Each child takes up 50% of the container */
    padding-left: 20px; /* Adds some padding to the left side for spacing */
  }

  .features-container.featuresLeft.feature-button{
    font-size: 40px;
    margin-right: 40px;
    padding: 80px 80px;
    background: linear-gradient(145deg, rgba(65, 184, 131, 0.85), rgba(88, 86, 214, 0.85));
    border: medium;
    border-radius: 8px;
  }
  .features-container.featuresRight .feature-button {
    font-size: 40px;
    margin-left: 40px;
    padding: 80px 80px;
    background: linear-gradient(145deg, rgba(65, 184, 131, 0.85), rgba(88, 86, 214, 0.85));
    border: medium;
    border-radius: 8px;
  }

  .features-container .el-button {
    font-size: 40px; /* Adjust the font size for better visibility */
    padding: 80px 80px; /* Adjust padding for a bigger button */
    background: linear-gradient(145deg, rgba(65, 184, 131, 0.85), rgba(88, 86, 214, 0.85)); /* Modern gradient background */
    border: medium; /* Optional: Remove border if present */
    border-radius: 8px; /* Optional: Adjust border-radius for rounded corners */
  }


  @media (max-width: 768px) {
    .loginIn {
      flex-direction: column;
      justify-content: flex-start;
      padding: 20px;
      background-image: none; /* 移动端可能不需要复杂背景，以减少加载时间 */
      background-color: #ffffff; /* 或者选择一个适合的背景色 */
    }

    .features-container {
      flex-direction: column;
      height: auto;
      padding: 0;
      justify-content: space-around;
      align-items: center;
    }

    .featuresLeft, .featuresRight {
      margin: 10px 0;
      padding: 0;
    }

    .features-container .el-button {
      font-size: 26px;
      padding: 40px 40px;
    }

    .login-form, .title-container, .el-form-item {
      width: 100%;
      margin-bottom: 0px; /* 或者更小的值，根据实际效果调整 */
    }

    .title-container .title {
      font-size: 18px; /* 减小字体大小以适应较小屏幕 */
    }

    .el-form-item {
      margin-bottom: 10px; /* 减少表单项之间的间距 */
    }

    .el-input, .el-form-item, .el-button {
      font-size: 14px; /* 减小字体大小以适应较小屏幕 */
    }

    .el-button {
      margin-top: 10px; /* 为按钮添加更多间距 */
    }

    .setting .register {
      font-size: 12px; /* 可能需要减小字体大小以适应屏幕 */
    }

    .confirm-button {
      margin-top: 15px;
    }
  }

}
</style>