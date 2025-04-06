<template>
  <div>
    <div class="container">
      <div class="login-form" style="backgroundColor:rgba(255, 255, 255, 0.26);borderRadius:10px">
        <h1 class="h1" style="color:rgb(9,9,9);fontSize:28px;">电池管理系统注册</h1>
    <el-form :model="ruleForm" :rules="rules" ref="ruleForm" class="rgs-form" label-width="120px">

			<!--prop属性用来关联表单项与验证规则-->
      <el-form-item label="用户名" prop="username" class="input">
        <el-input v-model="ruleForm.username" autocomplete="off" placeholder="用户名" />
      </el-form-item>
      <el-form-item label="密码" prop="password" class="input">
        <el-input v-model="ruleForm.password" type="password" autocomplete="off" placeholder="密码" />
      </el-form-item>
      <el-form-item label="邮箱" prop="email" class="input">
        <el-input v-model="ruleForm.email" autocomplete="off" placeholder="邮箱" />
      </el-form-item>


      <el-form-item class="input">
        <el-button class="btn" type="primary" @click="register()">注册</el-button>
        <el-button class="btn" type="default" @click="goBack()">返回</el-button>
      </el-form-item>

		</el-form>
      </div>
    </div>
  </div>
</template>
<script>
export default {
  data() {
    return {
      ruleForm: {
        username:"",
        password:"",
        email:"",
      },
      rules: {
        username: [
          { required: true, message: '请输入用户名', trigger: 'blur' }
        ],
        password: [
          { required: true, message: '请输入密码', trigger: 'blur' }
        ],
        email: [
          { required: true, message: '请输入邮箱地址', trigger: 'blur' },
          { type: 'email', message: '请输入正确的邮箱地址', trigger: ['blur', 'change'] }
        ]
      },

    };
  },
  mounted(){
    let table = this.$storage.get("loginTable");
  },
  methods: {
    // 获取uuid
    getUUID () {
      return new Date().getTime();
    },

    register() {
      this.$refs['ruleForm'].validate((valid) => {
        if (valid) {
          // 如果表单验证通过，则进行注册逻辑
          let registDto = {
            username: this.ruleForm.username,
            password: this.ruleForm.password,
            email: this.ruleForm.email
          };

          this.$http.post('/users/register', registDto)
              .then(response => {
                // 处理注册成功逻辑
                console.log(response);
                if(response.data.status===0){
                  this.$message.success('注册成功');
                }else{
                  this.$message.error(response.data.message);
                }
              })
              .catch(error => {
                // 处理错误情况
                console.error(error);
                this.$message.error('注册失败');
              });

        } else {
          console.log('error submit!!');
          this.$message.error("请填入完整信息");
          return false;
        }
      });
    },

    goBack() {
      // 实现返回按钮的逻辑，这里可以简单地返回上一页
      this.$router.go(-1);
    }

  }
};
</script>

<style lang="scss" scoped>



	.el-radio__input.is-checked .el-radio__inner {
		border-color: #00c292;
		background: #00c292;
	}

	.el-radio__input.is-checked .el-radio__inner {
		border-color: #00c292;
		background: #00c292;
	}

	.el-radio__input.is-checked .el-radio__inner {
		border-color: #00c292;
		background: #00c292;
	}

	.el-radio__input.is-checked+.el-radio__label {
		color: #00c292;
	}

	.el-radio__input.is-checked+.el-radio__label {
		color: #00c292;
	}

	.el-radio__input.is-checked+.el-radio__label {
		color: #00c292;
	}

	.h1 {
		margin-top: 10px;
	}

	body {
		padding: 0;
		margin: 0;
	}

	.nk-navigation {
		margin-top: 15px;

		a {
			display: inline-block;
			color: #fff;
			background: rgba(255, 255, 255, .2);
			width: 100px;
			height: 50px;
			border-radius: 30px;
			text-align: center;
			display: flex;
			align-items: center;
			margin: 0 auto;
			justify-content: center;
			padding: 0 20px;
		}

		.icon {
			margin-left: 10px;
			width: 30px;
			height: 30px;
		}
	}

	.register-container {
		margin-top: 10px;

		a {
			display: inline-block;
			color: #fff;
			max-width: 500px;
			height: 50px;
			border-radius: 30px;
			text-align: center;
			display: flex;
			align-items: center;
			margin: 0 auto;
			justify-content: center;
			padding: 0 20px;

			div {
				margin-left: 10px;
			}
		}
	}

	.container {
    background-image: url('../assets/img/loginBG.png');
		height: 100vh;
		background-position: center center;
		background-size: cover;
		background-repeat: no-repeat;

		.login-form {
			right: 50%;
			top: 50%;
			height: auto;
			transform: translate3d(50%, -50%, 0);
			border-radius: 10px;
			background-color: rgba(255,255,255,.5);
			width: 420px;
			padding: 30px 30px 40px 30px;
			font-size: 14px;
			font-weight: 500;

			.h1 {
				margin: 0;
				text-align: center;
				line-height: 54px;
			    font-size: 24px;
			    color: #000;
			}

			.rgs-form {
				display: flex;
				flex-direction: column;
				justify-content: center;
				align-items: center;

				.input {
					width: 100%;

					& /deep/ .el-form-item__label {
						line-height: 40px;
						color: rgb(30, 29, 29);
						font-size: #17181a;
					}

					& /deep/ .el-input__inner {
						height: 40px;
						color: #606266;
						font-size: 14px;
						border-width: 1px;
						border-style: solid;
						border-color: #606266;
						border-radius: 4px;
						background-color: #fff;
					}
				}

				.btn {
					width: 88px;
					height: 44px;
					color: rgba(255, 255, 255, 1);
					font-size: 14px;
					border-width: 1px;
					border-style: solid;
					border-color: rgba(117, 113, 249, 1);
					border-radius: 4px;
					background-color: rgba(117, 113, 249, 1);
				}
			}
		}
	}
</style>
