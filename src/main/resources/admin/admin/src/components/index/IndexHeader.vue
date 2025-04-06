<template>
	<div class="navbar" :style="{backgroundColor:heads.headBgColor,height:heads.headHeight,boxShadow:heads.headBoxShadow,lineHeight:heads.headHeight}">
		<div class="title-menu" :style="{justifyContent:heads.headTitleStyle=='1'?'flex-start':'center'}">
      <img src="../../assets/img/logohead.png" style="width: 80px; height: 80px;margin-right: 80px;">
<!--      <div class="divider"></div>  &lt;!&ndash; 紫色分割线 &ndash;&gt;-->
<!--      <img src="../../assets/img/logohead2.jpg" style="width: 80px; height: 80px;  margin-right: 20px;">-->
      <div class="title-name" :style="{color:heads.headFontColor,fontSize:heads.headFontSize}">{{this.$project.projectName}}</div>
		</div>
		<div class="right-menu">
			<div class="user-info" :style="{color:heads.headUserInfoFontColor,fontSize:heads.headUserInfoFontSize}">{{this.$storage.get('role')}} {{this.$storage.get('adminName')}}</div>
<!--			<div class="logout" :style="{color:heads.headLogoutFontColor,fontSize:heads.headLogoutFontSize}" @click="onIndexTap">退出到前台</div>-->
			<div class="logout" :style="{color:heads.headLogoutFontColor,fontSize:heads.headLogoutFontSize}" @click="onLogout">退出登录</div>
		</div>
	</div>
</template>

<script>
	export default {
		data() {
			return {
				dialogVisible: false,
				ruleForm: {},
				user: {},
				heads: {"headLogoutFontHoverColor":"#fff","headFontSize":"25px","headUserInfoFontColor":"rgba(255, 255, 255, 1)","headBoxShadow":"0 1px 6px #444","headTitleImgHeight":"44px","headLogoutFontHoverBgColor":"rgba(36, 38, 42, 1)","headFontColor":"rgba(255, 255, 255, 1)","headTitleImg":false,"headHeight":"80px","headTitleImgBorderRadius":"22px","headTitleImgUrl":"http://codegen.caihongy.cn/20201021/cc7d45d9c8164b58b18351764eba9be1.jpg","headBgColor":"#000000","headTitleImgBoxShadow":"0 1px 6px #444","headLogoutFontColor":"rgba(255, 255, 255, 1)","headUserInfoFontSize":"20px","headTitleImgWidth":"44px","headTitleStyle":"1","headLogoutFontSize":"16px"},
			};
		},
		created() {
			this.setHeaderStyle()
		},
		mounted() {
			let sessionTable = this.$storage.get("sessionTable")
			this.$http({
				url: sessionTable + '/session',
				method: "get"
			}).then(({
				data
			}) => {
				if (data && data.code === 0) {
					this.user = data.data;
				} else {
					let message = this.$message
					message.error(data.msg);
				}
			});
		},
		methods: {
			onLogout() {
				let storage = this.$storage
				let router = this.$router
				storage.remove("Token");
				router.replace({
					name: "login"
				});
			},
      onIndexTap(){
        window.location.href = `${this.$base.indexUrl}`
      },
			setHeaderStyle() {
			  this.$nextTick(()=>{
			    document.querySelectorAll('.navbar .right-menu .logout').forEach(el=>{
			      el.addEventListener("mouseenter", e => {
			        e.stopPropagation()
			        el.style.backgroundColor = this.heads.headLogoutFontHoverBgColor
					el.style.color = this.heads.headLogoutFontHoverColor
			      })
			      el.addEventListener("mouseleave", e => {
			        e.stopPropagation()
			        el.style.backgroundColor = "transparent"
					el.style.color = this.heads.headLogoutFontColor
			      })
			    })
			  })
			},
		}
	};
</script>


<style lang="scss" scoped>
	.navbar {
		height: 60px;
		line-height: 60px;
		width: 100%;
		padding: 0 34px;
		box-sizing: border-box;
		background-color: #ff00ff;
		position: relative;
		z-index: 111;
		
		.right-menu {
			position: absolute;
			right: 34px;
			top: 0;
			height: 100%;
			display: flex;
			justify-content: flex-end;
			align-items: center;
			z-index: 111;
			
			.user-info {
				font-size: 16px;
				color: red;
				padding: 0 12px;
			}
			
			.logout {
				font-size: 16px;
				color: red;
				padding: 0 12px;
				cursor: pointer;
			}
			
		}

    .divider {
      width: 2px; // 线的宽度
      height: 50px; // 线的高度
      background-color: #8a2be2; // 紫色
      margin: 0 20px; // 左右两侧各20px间隙
    }
		
		.title-menu {
			display: flex;
			justify-content: flex-start;
			align-items: center;
			width: 100%;
			height: 100%;
			
			.title-img {
				width: 44px;
				height: 44px;
				border-radius: 22px;
				box-shadow: 0 1px 6px #444;
				margin-right: 16px;
			}
			
			.title-name {
				font-size: 24px;
				color: #fff;
				font-weight: 700;
			}
		}
	}


  /* 移动端样式调整 */
  @media (max-width: 768px) {
    .navbar {
      flex-direction: column;
      justify-content: center;
      padding: 10px 10px;
      height: auto;

      .title-menu {
        justify-content: center;

        .title-img {
          display: none; /* 可以选择隐藏图片以节省空间 */
        }

        .title-name {
          font-size: 18px; /* 减小字体尺寸以适应屏幕 */
          margin-bottom: 5px; /* 为标题和下面的用户信息/退出按钮添加间距 */
        }
      }

      .right-menu {
        justify-content: center;

        .user-info, .logout {
          font-size: 14px; /* 进一步减小字体尺寸 */
          padding: 2px 5px; /* 减少内边距 */
        }

        .logout {
          border: 1px solid #FFFFFF; /* 为退出登录按钮添加边框，增加可点击感 */
          cursor: pointer;
          border-radius: 5px; /* 圆角 */
          margin-top: 5px; /* 与用户信息间隔 */
        }
      }
    }
  }
</style>
