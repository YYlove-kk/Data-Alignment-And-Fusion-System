import { createRouter, createWebHistory } from 'vue-router'
import Homepage from '../views/Homepage.vue'
import DataUpload from '../views/DataUpload.vue'
import DataAlignment from '../views/DataAlignment.vue'
import MultiModalKnowledgeGraph from '../views/MultiModalKnowledgeGraph.vue'
import PatientDataAnalysis from '../views/PatientDataAnalysis.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'home',
      component: Homepage
    },
    
    {
      path: "/dataUpload",
      name: "dataUpload",
      component: DataUpload
    },
    {
      path: "/dataAlignment",
      name: "dataAlignment",
      component: DataAlignment
    },
    {
      path: "/multiModalKnowledgeGraph",
      name: "multiModalKnowledgeGraph",
      component: MultiModalKnowledgeGraph
    },
    {
      path: "/patientDataAnalysis",
      name: "patientDataAnalysis",
      component: PatientDataAnalysis
    },
    // 其他路由配置...
  ]
})

export default router 