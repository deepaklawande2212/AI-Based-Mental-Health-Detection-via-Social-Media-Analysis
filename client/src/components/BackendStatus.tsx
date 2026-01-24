// /**
//  * Backend Status Component
//  * Shows the connection status to the FastAPI backend
//  */

// import { useState, useEffect } from 'react'
// import { CheckCircle, XCircle, RefreshCw } from 'lucide-react'
// import { useHealthCheck } from '../hooks/useApi'

// const BackendStatus = () => {
//   const { health, loading, error, checkHealth } = useHealthCheck()
//   const [lastChecked, setLastChecked] = useState<Date | null>(null)

//   // Check health on component mount
//   useEffect(() => {
//     checkHealth()
//   }, [checkHealth])

//   useEffect(() => {
//     if (health || error) {
//       setLastChecked(new Date())
//     }
//   }, [health, error])

//   const getStatusIcon = () => {
//     if (loading) {
//       return <RefreshCw className="h-4 w-4 animate-spin text-blue-500" />
//     }
//     if (health && health.status === 'healthy') {
//       return <CheckCircle className="h-4 w-4 text-green-500" />
//     }
//     return <XCircle className="h-4 w-4 text-red-500" />
//   }

//   const getStatusText = () => {
//     if (loading) return 'Checking...'
//     if (health && health.status === 'healthy') return 'Connected'
//     if (error) return 'Disconnected'
//     return 'Unknown'
//   }

//   const getStatusColor = () => {
//     if (loading) return 'text-blue-600 bg-blue-50 border-blue-200'
//     if (health && health.status === 'healthy') return 'text-green-600 bg-green-50 border-green-200'
//     return 'text-red-600 bg-red-50 border-red-200'
//   }

//   return (
//     <div className={`inline-flex items-center px-3 py-1 rounded-full border text-xs font-medium ${getStatusColor()}`}>
//       {getStatusIcon()}
//       <span className="ml-1">Backend: {getStatusText()}</span>
//       {lastChecked && (
//         <span className="ml-2 opacity-75">
//           {lastChecked.toLocaleTimeString()}
//         </span>
//       )}
//       <button
//         onClick={checkHealth}
//         className="ml-2 hover:opacity-75"
//         title="Refresh status"
//       >
//         <RefreshCw className="h-3 w-3" />
//       </button>
//     </div>
//   )
// }

// export default BackendStatus

/**
 * Backend Status Component (Temporarily disabled for demo)
 */

const BackendStatus = () => {
  // For presentation/demo only — hiding backend connection status
  return null;
};

export default BackendStatus;
