const path = require('path')

function resolve(dir) {
    return path.join(__dirname, dir)
}

function publicPath(){
    if (process.env.NODE_ENV === 'production') {
        return "././";
    } else {
        return "/";
    }
}

module.exports = {
    publicPath: publicPath(),
    configureWebpack: {
        resolve: {
            alias: {
                '@': resolve('src')
            }
        }
    },
    lintOnSave: false,
    devServer: {
        host: "0.0.0.0",
        port: 8085, // 使用8085端口进行前端开发
        hot: true,
        https: false,
        proxy: {
            '/api': { // 假定前端发往/api的请求都需要代理到后端
                target: 'http://122.51.105.149:8080', // 后端服务的URL
                changeOrigin: true,
                pathRewrite: {
                    '^/api': '' // 重写URL路径，将/api替换为空，依据后端实际API路径配置
                }
            }
        }
    },
    chainWebpack(config) {
        config.module
            .rule('svg')
            .exclude.add(resolve('src/icons'))
            .end()
        config.module
            .rule('icons')
            .test(/\.svg$/)
            .include.add(resolve('src/icons'))
            .end()
            .use('svg-sprite-loader')
            .loader('svg-sprite-loader')
            .options({
                symbolId: 'icon-[name]'
            })
            .end()
    }
}