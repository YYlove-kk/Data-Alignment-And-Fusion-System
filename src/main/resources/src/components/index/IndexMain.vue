<template>
	<el-main>
		<bread-crumbs :title="title" class="bread-crumbs"></bread-crumbs>
		<router-view class="router-view"></router-view>
	</el-main>
</template>
<script>
	import menu from "@/utils/menu";
	export default {
		data() {
			return {
				menuList: [],
				role: "",
				currentIndex: -2,
				itemMenu: [],
				title: ''
			};
		},
		mounted() {
			let menus = menu.list();
			this.menuList = menus;
			this.role = this.$storage.get("role");
		},
		methods: {
			menuHandler(menu) {
				this.$router.push({
					name: menu.tableName
				});
				this.title = menu.menu;
			},
			titleChange(index, menus) {
				this.currentIndex = index
				this.itemMenu = menus;
				console.log(menus);
			},
			homeChange(index) {
				this.itemMenu = [];
				this.title = ""
				this.currentIndex = index
				this.$router.push({
					name: 'home'
				});
			},
			// centerChange(index) {
			// 	this.itemMenu = [{
			// 		"buttons": ["新增", "查看", "修改", "删除"],
			// 		"menu": "修改密码",
			// 		"tableName": "updatePassword"
			// 	}, {
			// 		"buttons": ["新增", "查看", "修改", "删除"],
			// 		"menu": "个人信息",
			// 		"tableName": "center"
			// 	}];
			// 	this.title = ""
			// 	this.currentIndex = index
			// 	this.$router.push({
			// 		name: 'home'
			// 	});
			// }
		}
	};
</script>
<style lang="scss" scoped>
	a {
		text-decoration: none;
		color: #555;
	}

	a:hover {
		background: #00c292;
	}

	.nav-list {
		width: 100%;
		margin: 0 auto;
		text-align: left;
		margin-top: 20px;

		.nav-title {
			display: inline-block;
			font-size: 15px;
			color: #333;
			padding: 15px 25px;
			border: none;
		}

		.nav-title.active {
			color: #555;
			cursor: default;
			background-color: #fff;
		}
	}

	.nav-item {
		margin-top: 20px;
		background: #FFFFFF;
		padding: 15px 0;

		.menu {
			padding: 15px 25px;
		}
	}

	.el-main {
		background-color: #F6F8FA;
		padding: 0 24px;
		// padding-top: 60px;
	}

	.router-view {
		padding: 10px;
		margin-top: 10px;
		background: #FFFFFF;
		box-sizing: border-box;
	}

	.bread-crumbs {
		width: 100%;
		// border-bottom: 1px solid #e9eef3;
		// border-top: 1px solid #e9eef3;
		margin-top: 10px;
		box-sizing: border-box;
	}


  /* 响应式布局适配 */
  @media (max-width: 768px) {
    .nav-title {
      font-size: 12px; /* 更小的屏幕上进一步减小字体大小 */
      padding: 8px 10px; /* 减小内边距 */
    }

    .nav-item .menu {
      padding: 8px 10px; /* 减小内边距 */
    }

    .el-main {
      padding: 0 10px; /* 减小内边距以最大化屏幕利用率 */
    }

    .router-view {
      margin-top: 8px; /* 调整间距 */
      padding: 8px; /* 减小内边距 */
    }

    .bread-crumbs {
      padding: 8px 0; /* 减小内边距 */
    }
  }

</style>
