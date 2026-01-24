/**
 * Utility functions for transforming API responses
 * to component-friendly formats
 */

import { AnalysisResult as ApiAnalysisResult } from "../services/api";

// Component expected format
export interface ComponentAnalysisResult {
  sentiment: {
    positive: number;
    negative: number;
    neutral: number;
  };
  riskLevel: "low" | "medium" | "high";
  emotions: {
    joy: number;
    sadness: number;
    anger: number;
    fear: number;
    surprise: number;
    disgust: number;
  };
  confidence: number;
  recommendations: string[];
  modelResults?: any[]; // Add model results for detailed view
  analyzedContent?: string[]; // Add analyzed content for detailed view
  contentSummary?: string; // Add content summary for detailed view
}

/**
 * Transform API analysis result to component format
 */
export const transformAnalysisResult = (
  apiResult: ApiAnalysisResult
): ComponentAnalysisResult => {
  // If we have model_results, aggregate them to get the best results
  if (apiResult.model_results && apiResult.model_results.length > 0) {
    // Use the backend's best model selection if available
    let bestModel = apiResult.model_results[0];

    if (apiResult.best_model) {
      // Find the model that the backend selected as best
      const backendBestModel = apiResult.model_results.find(
        (model) => model.model_name === apiResult.best_model
      );
      if (backendBestModel) {
        bestModel = backendBestModel;
      }
    } else {
      // Fallback: Find the best performing model, but avoid artificially high confidence (1.0)
      // Prefer BERT if available, otherwise use the most realistic confidence
      const bertModel = apiResult.model_results.find(
        (model) => model.model_name === "bert"
      );
      if (bertModel && bertModel.confidence > 0.5) {
        bestModel = bertModel;
      } else {
        // Find model with highest realistic confidence (not 1.0)
        bestModel = apiResult.model_results.reduce((best, current) => {
          // Avoid models with artificially high confidence (1.0)
          if (current.confidence === 1.0) return best;
          if (best.confidence === 1.0) return current;

          return current.confidence > best.confidence ? current : best;
        });
      }
    }

    // Use the best model's sentiment data
    const sentiment = bestModel.sentiment || {
      positive: 0,
      negative: 0,
      neutral: 0,
      overall: "neutral",
      confidence: 0,
    };

    // Prioritize final_emotions over individual model emotions (final_emotions has real data)
    const emotions = apiResult.final_emotions ||
      bestModel.emotions || {
        joy: 0,
        sadness: 0,
        anger: 0,
        fear: 0,
        surprise: 0,
        disgust: 0,
        dominant_emotion: "neutral",
        confidence: 0,
      };

    // Use the best model's risk assessment
    const risk = bestModel.risk || {
      level: "low",
      score: 0,
      factors: [],
      recommendations: [],
      confidence: 0,
    };

    // Map risk level
    const riskLevel = risk.level.toLowerCase() as "low" | "medium" | "high";

    return {
      sentiment: {
        positive: sentiment.positive,
        negative: sentiment.negative,
        neutral: sentiment.neutral,
      },
      riskLevel,
      emotions: {
        joy: emotions.joy,
        sadness: emotions.sadness,
        anger: emotions.anger,
        fear: emotions.fear,
        surprise: emotions.surprise,
        disgust: emotions.disgust,
      },
      confidence: (apiResult.confidence || 0.75) * 100,
      recommendations: risk.recommendations || ["Continue monitoring"],
      modelResults: apiResult.model_results,
      analyzedContent: apiResult.analyzed_content, // Include analyzed content
      contentSummary: apiResult.content_summary, // Include content summary
    };
  }

  // Fallback for when no model results are available
  return {
    sentiment: {
      positive: 0,
      negative: 0,
      neutral: 0,
    },
    riskLevel: "low",
    emotions: {
      joy: 0,
      sadness: 0,
      anger: 0,
      fear: 0,
      surprise: 0,
      disgust: 0,
    },
    confidence: 0,
    recommendations: ["No analysis data available"],
    analyzedContent: apiResult.analyzed_content,
    contentSummary: apiResult.content_summary,
  };
};

/**
 * Format error messages for display
 */
export const formatErrorMessage = (error: string): string => {
  if (error.includes("fetch")) {
    return "Unable to connect to the server. Please check your connection.";
  }
  if (error.includes("401") || error.includes("unauthorized")) {
    return "Authentication failed. Please check your API credentials.";
  }
  if (error.includes("404")) {
    return "Resource not found. Please check the request parameters.";
  }
  if (error.includes("500")) {
    return "Server error occurred. Please try again later.";
  }
  return error;
};

/**
 * Get risk level color classes
 */
export const getRiskLevelClasses = (level: string) => {
  switch (level.toLowerCase()) {
    case "low":
      return "text-green-600 bg-green-50 border-green-200";
    case "medium":
      return "text-yellow-600 bg-yellow-50 border-yellow-200";
    case "high":
      return "text-red-600 bg-red-50 border-red-200";
    default:
      return "text-gray-600 bg-gray-50 border-gray-200";
  }
};

/**
 * Format analysis status for display
 */
export const formatAnalysisStatus = (status: string): string => {
  switch (status.toLowerCase()) {
    case "pending":
      return "Analysis in progress...";
    case "processing":
      return "Processing data...";
    case "completed":
      return "Analysis completed";
    case "failed":
      return "Analysis failed";
    default:
      return status;
  }
};

/**
 * Calculate overall sentiment
 */
export const calculateOverallSentiment = (sentiment: {
  positive: number;
  negative: number;
  neutral: number;
}): string => {
  const { positive, negative, neutral } = sentiment;

  if (positive > negative && positive > neutral) {
    return "Positive";
  } else if (negative > positive && negative > neutral) {
    return "Negative";
  } else {
    return "Neutral";
  }
};

/**
 * Format processing time
 */
export const formatProcessingTime = (seconds: number): string => {
  if (seconds < 60) {
    return `${Math.round(seconds)}s`;
  } else {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.round(seconds % 60);
    return `${minutes}m ${remainingSeconds}s`;
  }
};

/**
 * Validate Twitter username
 */
export const validateTwitterUsername = (username: string): string | null => {
  const cleanUsername = username.replace("@", "").trim();

  if (!cleanUsername) {
    return "Username cannot be empty";
  }

  if (cleanUsername.length > 15) {
    return "Username cannot be longer than 15 characters";
  }

  if (!/^[a-zA-Z0-9_]+$/.test(cleanUsername)) {
    return "Username can only contain letters, numbers, and underscores";
  }

  return null;
};

/**
 * Validate CSV file
 */
export const validateCSVFile = (file: File): string | null => {
  if (!file) {
    return "No file selected";
  }

  if (file.type !== "text/csv" && !file.name.endsWith(".csv")) {
    return "File must be a CSV format";
  }

  if (file.size > 10 * 1024 * 1024) {
    // 10MB
    return "File size must be less than 10MB";
  }

  return null;
};
