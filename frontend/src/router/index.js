import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '../views/HomeView.vue'
import ReliableView from '../views/ReliableView.vue'
import UnreliableView from '../views/UnreliableView.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'home',
      component: HomeView
    },
    {
      path: '/reliable',
      name: 'reliable',
      component: ReliableView
    },
    {
      path: '/unreliable',
      name: 'unreliable',
      component: UnreliableView
    }
  ]
})

export default router

