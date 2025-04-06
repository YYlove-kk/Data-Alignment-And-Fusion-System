const base = {
    refreshTime: 300000,  // 默认5分钟，以毫秒为单位

    get() {
                return {
            // url : "http://122.51.105.149:8080/",
            url : "http://localhost:8080/",
            name: "springbootdtjr3",
            // 退出到首页链接
            // indexUrl: 'http://122.51.105.149:8080/springbootdtjr3/front/index.html'
            indexUrl: 'http://localhost:8080/springbootdtjr3/front/index.html'
        };
            },
    getProjectName(){
        return {
            projectName: "电池管理系统"
        } 
    },

    getRefreshTime() {
        return localStorage.getItem('refreshTime') || 300000;
    },

    // 设置刷新时间并保存到localStorage
    setRefreshTime(minutes) {
        const milliseconds = minutes * 60 * 1000;
        localStorage.setItem('refreshTime', milliseconds.toString());
    },

}
export default base
