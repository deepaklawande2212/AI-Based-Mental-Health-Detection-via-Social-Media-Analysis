// API base URL - update this when you have your backend server
export const API_BASE_URL = 'http://localhost:8000/api'

// API endpoints
export const API_ENDPOINTS = {
  ANALYZE_TEXT: '/analyze/text',
  ANALYZE_CSV: '/analyze/csv',
  ANALYZE_TWITTER: '/analyze/twitter',
  HEALTH_CHECK: '/health'
}

// Analysis types
export const ANALYSIS_TYPES = {
  CSV: 'csv',
  TWITTER: 'twitter'
} as const

// Risk levels
export const RISK_LEVELS = {
  LOW: 'low',
  MEDIUM: 'medium',
  HIGH: 'high'
} as const

// Emotion types
export const EMOTIONS = {
  JOY: 'joy',
  SADNESS: 'sadness',
  ANGER: 'anger',
  FEAR: 'fear',
  SURPRISE: 'surprise',
  DISGUST: 'disgust'
} as const

// AI Models
export const AI_MODELS = {
  CNN: {
    name: 'CNN',
    fullName: 'Convolutional Neural Network',
    description: 'Pattern recognition and feature extraction'
  },
  DNN: {
    name: 'DNN',
    fullName: 'Deep Neural Network',
    description: 'Multi-layered deep learning analysis'
  },
  MOON: {
    name: 'MOON',
    fullName: 'Mood-Oriented Neural',
    description: 'Specialized mood detection'
  },
  CASTLE: {
    name: 'CASTLE',
    fullName: 'Comprehensive Assessment System',
    description: 'Holistic mental health evaluation'
  }
} as const
