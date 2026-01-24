export const formatPercentage = (value: number): string => {
  return `${Math.round(value)}%`
}

export const formatConfidence = (value: number): string => {
  return `${Math.round(value * 100) / 100}%`
}

export const capitalizeFirst = (str: string): string => {
  return str.charAt(0).toUpperCase() + str.slice(1).toLowerCase()
}

export const formatTimestamp = (timestamp: string): string => {
  return new Date(timestamp).toLocaleString()
}

export const getRiskColor = (riskLevel: string): string => {
  switch (riskLevel) {
    case 'low':
      return 'text-green-600'
    case 'medium':
      return 'text-yellow-600'
    case 'high':
      return 'text-red-600'
    default:
      return 'text-gray-600'
  }
}

export const getRiskBgColor = (riskLevel: string): string => {
  switch (riskLevel) {
    case 'low':
      return 'bg-green-50 border-green-200'
    case 'medium':
      return 'bg-yellow-50 border-yellow-200'
    case 'high':
      return 'bg-red-50 border-red-200'
    default:
      return 'bg-gray-50 border-gray-200'
  }
}

export const getEmotionColor = (emotion: string): string => {
  const colors: { [key: string]: string } = {
    joy: '#fbbf24',
    sadness: '#3b82f6',
    anger: '#ef4444',
    fear: '#8b5cf6',
    surprise: '#f59e0b',
    disgust: '#059669'
  }
  return colors[emotion.toLowerCase()] || '#6b7280'
}

export const validateEmail = (email: string): boolean => {
  const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
  return re.test(email)
}

export const validateTwitterHandle = (handle: string): boolean => {
  // Remove @ if present and check if valid username
  const cleanHandle = handle.replace('@', '')
  const re = /^[A-Za-z0-9_]{1,15}$/
  return re.test(cleanHandle)
}

export const downloadJSON = (data: any, filename: string): void => {
  const dataStr = JSON.stringify(data, null, 2)
  const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr)
  
  const exportFileDefaultName = `${filename}-${Date.now()}.json`
  
  const linkElement = document.createElement('a')
  linkElement.setAttribute('href', dataUri)
  linkElement.setAttribute('download', exportFileDefaultName)
  linkElement.click()
}

export const downloadCSV = (data: any[], filename: string): void => {
  if (data.length === 0) return
  
  const headers = Object.keys(data[0])
  const csvContent = [
    headers.join(','),
    ...data.map(row => headers.map(header => `"${row[header]}"`).join(','))
  ].join('\n')
  
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' })
  const link = document.createElement('a')
  
  if (link.download !== undefined) {
    const url = URL.createObjectURL(blob)
    link.setAttribute('href', url)
    link.setAttribute('download', `${filename}-${Date.now()}.csv`)
    link.style.visibility = 'hidden'
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }
}
