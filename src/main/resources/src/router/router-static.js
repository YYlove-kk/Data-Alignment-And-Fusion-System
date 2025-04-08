import Vue from 'vue';
import VueRouter from 'vue-router';

Vue.use(VueRouter);

// 导入组件
import Index from '@/views/index';
import Home from '@/views/home';
import Login from '@/views/login';
import NotFound from '@/views/404';
import UpdatePassword from '@/views/update-password';
import Pay from '@/views/pay';
import Register from '@/views/register';
import Center from '@/views/center';
import Device from '@/views/modules/device/list';
import BatteryModule from '@/views/modules/BatteryModule/list';
import ExceptionBatteryModule from '@/views/modules/ExceptionBatteryModule/list';
import DivideControl from '@/views/modules/DivideControl/list';
import Stock from '@/views/modules/Stock/list';
import TaskModule from '@/views/modules/TaskModule/adminTaskManage.vue';
import adminTaskManage from "@/views/modules/TaskModule/adminTaskManage.vue";
import userTask from "@/views/modules/TaskModule/userTask.vue";

// 配置路由
const routes = [
    {
        path: '/index',
        name: '首页',
        component: Index,
        children: [
            {
                path: '/',
                name: '首页',
                component: Home,
                meta: { icon: '', title: 'center' }
            },
            {
                path: '/diagnose',
                name: '首页',
                component: Home,
                meta: { icon: '', title: 'center' }
            },
            {
                path: '/updatePassword',
                name: '修改密码',
                component: UpdatePassword,
                meta: { icon: '', title: 'updatePassword' }
            },
            {
                path: '/pay',
                name: '支付',
                component: Pay,
                meta: { icon: '', title: 'pay' }
            },
            {
                path: '/center',
                name: '个人信息',
                component: Center,
                meta: { icon: '', title: 'center' }
            },
            {
                path: '/device',
                name: '设备',
                component: Device
            },
            {
                path: '/BatteryModule',
                name: '模组状态',
                component: BatteryModule
            },
            {
                path: '/ExceptionBatteryModule',
                name: '异常模组状态',
                component: ExceptionBatteryModule
            },
            {
                path: '/DivideControl',
                name: '分区管理',
                component: DivideControl
            },
            {
                path: '/Stock',
                name: '未处理模组',
                component: Stock
            }
        ]
    },
    {
        path: '/login',
        name: 'login',
        component: Login,
        meta: { icon: '', title: 'login' }
    },
    {
        path: '/BatteryModule',
        name: '模组状态',
        component: BatteryModule,
        props: true // 允许组件接收路由参数作为props
    },
    {
        path: '/ExceptionBatteryModule',
        name: '异常模组状态',
        component: ExceptionBatteryModule,
        props: true // 允许组件接收路由参数作为props
    },
    {
        path: '/adminTaskManage',
        name: '升级任务管理',
        component: adminTaskManage,
        props: true // 允许组件接收路由参数作为props
    },
    {
        path: '/userTask',
        name: '用户任务上传',
        component: userTask,
        props: true // 允许组件接收路由参数作为props
    },
    {
        path: '/TaskModule',
        name: '任务下发管理',
        component: TaskModule
    },
    {
        path: '/register',
        name: 'register',
        component: Register,
        meta: { icon: '', title: 'register' }
    },
    {
        path: '/',
        name: '首页',
        redirect: '/login'
    }, // 默认跳转路由
    {
        path: '*',
        component: NotFound
    }
];

const router = new VueRouter({
    mode: 'hash', // hash模式改为history
    routes // (缩写) 相当于 routes: routes
});

export default router;
