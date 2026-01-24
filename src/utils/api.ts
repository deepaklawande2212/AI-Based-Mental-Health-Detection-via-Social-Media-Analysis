import axios from 'axios'
import { API_BASE_URL } from '../constants'

// Create axios instance
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // 30 seconds for analysis requests
  headers: {
    'Content-Type': 'application/json'
  }
})

// Request interceptor
api.interceptors.request.use(
  (config) => {
    // Add any auth headers here if needed
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Response interceptor
api.interceptors.response.use(
  (response) => {
    return response
  },
  (error) => {
    // Handle common errors
    if (error.response?.status === 429) {
      throw new Error('Too many requests. Please try again later.')
    }
    if (error.response?.status === 500) {
      throw new Error('Server error. Please try again later.')
    }
    if (error.code === 'ECONNABORTED') {
      throw new Error('Request timeout. Please try again.')
    }
    return Promise.reject(error)
  }
)

export default api
