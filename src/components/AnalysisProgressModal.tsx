import React from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Brain,
  CheckCircle,
  Clock,
  Loader2,
  FileText,
  AlertTriangle,
} from "lucide-react";

interface AnalysisProgressModalProps {
  isOpen: boolean;
  analysisMode: "csv" | "twitter";
  currentStep: string;
  progress: number;
  modelStatus: {
    [key: string]: "pending" | "loading" | "completed" | "error";
  };
  onClose?: () => void;
}

const AnalysisProgressModal: React.FC<AnalysisProgressModalProps> = ({
  isOpen,
  analysisMode,
  currentStep,
  progress,
  modelStatus,
  onClose,
}) => {
  const models = [
    {
      name: "Decision Tree",
      key: "decision_tree",
      description: "Rule-based analysis",
    },
    { name: "CNN", key: "cnn", description: "Pattern detection" },
    { name: "LSTM", key: "lstm", description: "Temporal analysis" },
    { name: "RNN", key: "rnn", description: "Frequency analysis" },
    { name: "BERT", key: "bert", description: "Advanced language model" },
  ];

  const getStepIcon = (step: string) => {
    switch (step) {
      case "uploading":
        return <FileText className="h-5 w-5" />;
      case "preprocessing":
        return <Loader2 className="h-5 w-5 animate-spin" />;
      case "analyzing":
        return <Brain className="h-5 w-5" />;
      case "completing":
        return <CheckCircle className="h-5 w-5" />;
      default:
        return <Clock className="h-5 w-5" />;
    }
  };

  const getStepColor = (step: string) => {
    switch (step) {
      case "uploading":
        return "text-blue-600";
      case "preprocessing":
        return "text-yellow-600";
      case "analyzing":
        return "text-purple-600";
      case "completing":
        return "text-green-600";
      default:
        return "text-gray-600";
    }
  };

  const getModelStatusIcon = (status: string) => {
    switch (status) {
      case "pending":
        return <Clock className="h-4 w-4 text-gray-400" />;
      case "loading":
        return <Loader2 className="h-4 w-4 text-blue-500 animate-spin" />;
      case "completed":
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case "error":
        return <AlertTriangle className="h-4 w-4 text-red-500" />;
      default:
        return <Clock className="h-4 w-4 text-gray-400" />;
    }
  };

  const getModelStatusColor = (status: string) => {
    switch (status) {
      case "pending":
        return "text-gray-500";
      case "loading":
        return "text-blue-600";
      case "completed":
        return "text-green-600";
      case "error":
        return "text-red-600";
      default:
        return "text-gray-500";
    }
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50 p-4"
          onClick={onClose}
        >
          <motion.div
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.9, opacity: 0 }}
            className="bg-white rounded-xl shadow-2xl max-w-md w-full p-6"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div className="text-center mb-6">
              <div className="flex items-center justify-center mb-3">
                <Brain className="h-8 w-8 text-primary-600 mr-3" />
                <h2 className="text-xl font-bold text-gray-900">
                  Analysis in Progress
                </h2>
              </div>
              <p className="text-gray-600">
                {analysisMode === "csv"
                  ? "Processing your text/CSV file..."
                  : "Analyzing Twitter data..."}
              </p>
            </div>

            {/* Current Step */}
            <div className="mb-6">
              <div className="flex items-center justify-center mb-3">
                <div className={`mr-3 ${getStepColor(currentStep)}`}>
                  {getStepIcon(currentStep)}
                </div>
                <span className="text-sm font-medium text-gray-700 capitalize">
                  {currentStep.replace("_", " ")}
                </span>
              </div>

              {/* Progress Bar */}
              <div className="w-full bg-gray-200 rounded-full h-2 mb-2">
                <motion.div
                  className="bg-gradient-to-r from-blue-500 to-purple-600 h-2 rounded-full"
                  initial={{ width: 0 }}
                  animate={{ width: `${progress}%` }}
                  transition={{ duration: 0.5 }}
                />
              </div>
              <div className="text-center">
                <span className="text-sm text-gray-600">
                  {progress}% complete
                </span>
              </div>
            </div>

            {/* Model Status */}
            <div className="mb-6">
              <h3 className="text-sm font-medium text-gray-900 mb-3">
                AI Model Analysis
              </h3>
              <div className="space-y-2">
                {models.map((model) => (
                  <div
                    key={model.key}
                    className="flex items-center justify-between p-2 bg-gray-50 rounded-lg"
                  >
                    <div className="flex items-center">
                      {getModelStatusIcon(modelStatus[model.key] || "pending")}
                      <div className="ml-3">
                        <div className="text-sm font-medium text-gray-900">
                          {model.name}
                        </div>
                        <div className="text-xs text-gray-500">
                          {model.description}
                        </div>
                      </div>
                    </div>
                    <span
                      className={`text-xs font-medium capitalize ${getModelStatusColor(
                        modelStatus[model.key] || "pending"
                      )}`}
                    >
                      {modelStatus[model.key] || "pending"}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {/* Status Message */}
            <div className="text-center">
              <div className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-blue-100 text-blue-800">
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Please wait while we analyze your data...
              </div>
            </div>

            {/* Close Button (optional) */}
            {onClose && (
              <div className="mt-6 text-center">
                <button
                  onClick={onClose}
                  className="text-sm text-gray-500 hover:text-gray-700 transition-colors"
                >
                  Cancel Analysis
                </button>
              </div>
            )}
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default AnalysisProgressModal;
