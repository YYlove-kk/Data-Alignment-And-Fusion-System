import axios from 'axios'
import router from '@/router/router-static'
import storage from '@/utils/storage'
import base from "@/utils/base";

const http = axios.create({
    timeout: 1000 * 86400,
    withCredentials: true,
    // baseURL: "http://122.51.105.149:8080/",
    baseURL: "http://localhost:8080/",
    headers: {
        'Content-Type': 'application/json; charset=utf-8'
    }
})
// 请求拦截
http.interceptors.request.use(config => {
    config.headers['Token'] = storage.get('Token') // 请求头带上token
    return config
}, error => {
    return Promise.reject(error)
})
// 响应拦截
http.interceptors.response.use(response => {
    if (response.data && response.data.code === 401) { // 401, token失效
        router.push({ name: 'login' })
    }
    return response
}, error => {
    return Promise.reject(error)
})
export default http