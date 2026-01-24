/**
 * Custom hooks for API operations
 *
 * These hooks provide a clean interface for React components
 * to interact with the backend APIs with built-in state management.
 */

import { useState, useEffect, useCallback } from "react";
import {
  apiService,
  ApiResponse,
  AnalysisResult,
  TwitterAnalysisRequest,
  CSVAnalysisRequest,
  SimpleTextAnalysisRequest,
  TwitterData,
  CSVUploadResponse,
  HealthStatus,
} from "../services/api";

// Generic API hook for common patterns
export function useApi<T>() {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const execute = useCallback(
    async (apiCall: () => Promise<ApiResponse<T>>) => {
      setLoading(true);
      setError(null);

      try {
        const response = await apiCall();

        if (response.success && response.data) {
          setData(response.data);
          return response.data;
        } else {
          const errorMessage = response.error || "API call failed";
          setError(errorMessage);
          throw new Error(errorMessage);
        }
      } catch (err) {
        const errorMessage =
          err instanceof Error ? err.message : "Unknown error";
        setError(errorMessage);
        throw err;
      } finally {
        setLoading(false);
      }
    },
    []
  );

  const reset = useCallback(() => {
    setData(null);
    setError(null);
    setLoading(false);
  }, []);

  return { data, loading, error, execute, reset };
}

// Health check hook
export function useHealthCheck() {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const checkHealth = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await apiService.getHealth();

      if (response.success && response.data) {
        setHealth(response.data);
        return response.data;
      } else {
        const errorMessage = response.error || "Health check failed";
        setError(errorMessage);
        return null;
      }
    } catch (err) {
      const errorMessage =
        err instanceof Error ? err.message : "Health check error";
      setError(errorMessage);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    checkHealth();
  }, []);

  return { health, loading, error, checkHealth };
}

// Twitter analysis hook
export function useTwitterAnalysis() {
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(
    null
  );
  const [twitterData, setTwitterData] = useState<TwitterData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const analyzeTwitter = useCallback(
    async (request: TwitterAnalysisRequest) => {
      setLoading(true);
      setError(null);
      setAnalysisResult(null);
      setTwitterData(null);

      try {
        // Start the analysis
        const response = await apiService.analyzeTwitter(request);

        if (response.success && response.data) {
          setAnalysisResult(response.data);

          // Also fetch Twitter user data
          try {
            const userDataResponse = await apiService.getTwitterData(
              request.username
            );
            if (userDataResponse.success && userDataResponse.data) {
              setTwitterData(userDataResponse.data);
            }
          } catch (userErr) {
            console.warn("Could not fetch Twitter user data:", userErr);
          }

          return response.data;
        } else {
          const errorMessage = response.error || "Twitter analysis failed";
          setError(errorMessage);
          throw new Error(errorMessage);
        }
      } catch (err) {
        const errorMessage =
          err instanceof Error ? err.message : "Twitter analysis error";
        setError(errorMessage);
        throw err;
      } finally {
        setLoading(false);
      }
    },
    []
  );

  const reset = useCallback(() => {
    setAnalysisResult(null);
    setTwitterData(null);
    setError(null);
    setLoading(false);
  }, []);

  return {
    analysisResult,
    twitterData,
    loading,
    error,
    analyzeTwitter,
    reset,
  };
}

// CSV analysis hook
export function useCSVAnalysis() {
  const [uploadResponse, setUploadResponse] =
    useState<CSVUploadResponse | null>(null);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(
    null
  );
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const uploadCSV = useCallback(async (file: File) => {
    setLoading(true);
    setError(null);
    setUploadResponse(null);
    setAnalysisResult(null);

    try {
      const response = await apiService.uploadCSV(file);

      if (response.success && response.data) {
        setUploadResponse(response.data);
        return response.data;
      } else {
        const errorMessage = response.error || "CSV upload failed";
        setError(errorMessage);
        throw new Error(errorMessage);
      }
    } catch (err) {
      const errorMessage =
        err instanceof Error ? err.message : "CSV upload error";
      setError(errorMessage);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const analyzeCSV = useCallback(async (request: CSVAnalysisRequest) => {
    setLoading(true);
    setError(null);
    setAnalysisResult(null);

    try {
      const response = await apiService.analyzeCSV(request);

      if (response.success && response.data) {
        setAnalysisResult(response.data);
        return response.data;
      } else {
        const errorMessage = response.error || "CSV analysis failed";
        setError(errorMessage);
        throw new Error(errorMessage);
      }
    } catch (err) {
      const errorMessage =
        err instanceof Error ? err.message : "CSV analysis error";
      setError(errorMessage);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const reset = useCallback(() => {
    setUploadResponse(null);
    setAnalysisResult(null);
    setError(null);
    setLoading(false);
  }, []);

  return {
    uploadResponse,
    analysisResult,
    loading,
    error,
    uploadCSV,
    analyzeCSV,
    reset,
  };
}

// Analysis history hook
export function useAnalysisHistory() {
  const [analyses, setAnalyses] = useState<AnalysisResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchHistory = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await apiService.getAnalysisHistory();

      if (response.success && response.data) {
        setAnalyses(response.data);
        return response.data;
      } else {
        const errorMessage =
          response.error || "Failed to fetch analysis history";
        setError(errorMessage);
        return [];
      }
    } catch (err) {
      const errorMessage =
        err instanceof Error ? err.message : "Analysis history error";
      setError(errorMessage);
      return [];
    } finally {
      setLoading(false);
    }
  }, []);

  const deleteAnalysis = useCallback(async (analysisId: string) => {
    try {
      const response = await apiService.deleteAnalysis(analysisId);

      if (response.success) {
        setAnalyses((prev) =>
          prev.filter((analysis) => analysis.analysis_id !== analysisId)
        );
        return true;
      } else {
        setError(response.error || "Failed to delete analysis");
        return false;
      }
    } catch (err) {
      const errorMessage =
        err instanceof Error ? err.message : "Delete analysis error";
      setError(errorMessage);
      return false;
    }
  }, []);

  useEffect(() => {
    fetchHistory();
  }, [fetchHistory]);

  return {
    analyses,
    loading,
    error,
    fetchHistory,
    deleteAnalysis,
  };
}

// Polling hook for real-time analysis updates
export function useAnalysisPolling(
  analysisId: string | null,
  interval: number = 5000
) {
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [polling, setPolling] = useState(false);

  useEffect(() => {
    if (!analysisId) return;

    let timeoutId: number;

    const poll = async () => {
      try {
        const response = await apiService.getAnalysisResult(analysisId);

        if (response.success && response.data) {
          setResult(response.data);

          // Stop polling if analysis is complete
          if (
            response.data.status === "completed" ||
            response.data.status === "failed"
          ) {
            setPolling(false);
            return;
          }
        }

        // Continue polling
        timeoutId = window.setTimeout(poll, interval);
      } catch (err) {
        console.error("Polling error:", err);
        setPolling(false);
      }
    };

    setPolling(true);
    poll();

    return () => {
      if (timeoutId) {
        window.clearTimeout(timeoutId);
      }
      setPolling(false);
    };
  }, [analysisId, interval]);

  return { result, polling };
}

// Simple text analysis hook
export function useSimpleTextAnalysis() {
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(
    null
  );
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const analyzeText = useCallback(async (text: string) => {
    setLoading(true);
    setError(null);

    try {
      const response = await apiService.analyzeSimpleText({ text });

      if (response.success && response.data) {
        setAnalysisResult(response.data);
        return response.data;
      } else {
        const errorMessage = response.error || "Text analysis failed";
        setError(errorMessage);
        throw new Error(errorMessage);
      }
    } catch (err) {
      const errorMessage =
        err instanceof Error ? err.message : "Text analysis error";
      setError(errorMessage);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const reset = useCallback(() => {
    setAnalysisResult(null);
    setError(null);
    setLoading(false);
  }, []);

  return {
    analysisResult,
    loading,
    error,
    analyzeText,
    reset,
  };
}
