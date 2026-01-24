/**
 * API Service for Depression Detection System
 *
 * This service handles all communication with the FastAPI backend.
 * It provides typed interfaces for all API endpoints.
 */

const API_BASE_URL = "http://localhost:8000";

// API Response Types
export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  message?: string;
  error?: string;
}

export interface TwitterAnalysisRequest {
  username: string;
  max_tweets?: number;
  include_retweets?: boolean;
}

export interface CSVAnalysisRequest {
  file_content: string;
  filename: string;
  analysis_type?: string;
  text_column?: string;
}

export interface SimpleTextAnalysisRequest {
  text: string;
  analysis_type?: string;
}

export interface AnalysisResult {
  analysis_id: string;
  status: string;
  data_source: string;
  analysis_type: string;
  created_at: string;
  model_results: ModelResult[];
  final_sentiment?: SentimentScore;
  final_emotions?: EmotionScore;
  final_risk?: RiskAssessment;
  best_model?: string;
  processing_time: number;
  confidence: number;
  analyzed_content?: string[]; // Add analyzed content (tweets/text)
  content_summary?: string; // Add content summary
}

export interface ModelResult {
  model_name: string;
  accuracy: number;
  processing_time: number;
  confidence: number;
  status: string;
  sentiment?: SentimentScore;
  emotions?: EmotionScore;
  risk?: RiskAssessment;
}

export interface SentimentScore {
  positive: number;
  negative: number;
  neutral: number;
  overall: string;
  confidence: number;
}

export interface EmotionScore {
  joy: number;
  sadness: number;
  anger: number;
  fear: number;
  surprise: number;
  disgust: number;
  dominant_emotion: string;
  confidence: number;
}

export interface RiskAssessment {
  level: string;
  score: number;
  factors: string[];
  recommendations: string[];
  confidence: number;
}

export interface TwitterData {
  username: string;
  user_id?: string;
  display_name?: string;
  bio?: string;
  followers_count: number;
  following_count: number;
  tweet_count: number;
  tweets_collected: number;
  collection_status: string;
  created_at: string;
}

export interface CSVUploadResponse {
  file_id: string;
  filename: string;
  file_size: number;
  row_count: number;
  columns: string[];
  upload_date: string;
  status: string;
}

export interface HealthStatus {
  status: string;
  database: string;
  models: Record<string, boolean>;
  timestamp: string;
}

class ApiService {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  private async fetchApi<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<ApiResponse<T>> {
    try {
      const url = `${this.baseUrl}${endpoint}`;
      const response = await fetch(url, {
        headers: {
          "Content-Type": "application/json",
          ...options.headers,
        },
        ...options,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(
          errorData.message || `HTTP ${response.status}: ${response.statusText}`
        );
      }

      const data = await response.json();
      return {
        success: true,
        data,
      };
    } catch (error) {
      console.error(`API Error (${endpoint}):`, error);
      return {
        success: false,
        error:
          error instanceof Error ? error.message : "Unknown error occurred",
      };
    }
  }

  // Health Check
  async getHealth(): Promise<ApiResponse<HealthStatus>> {
    return this.fetchApi<HealthStatus>("/api/health/health");
  }

  // System Status
  async getSystemStatus(): Promise<ApiResponse<any>> {
    return this.fetchApi<any>("/api/health/status");
  }

  // Twitter Analysis APIs
  async analyzeTwitter(
    request: TwitterAnalysisRequest
  ): Promise<ApiResponse<AnalysisResult>> {
    return this.fetchApi<AnalysisResult>("/api/twitter/analyze-csv", {
      method: "POST",
      body: JSON.stringify(request),
    });
  }

  async getTwitterData(username: string): Promise<ApiResponse<TwitterData>> {
    return this.fetchApi<TwitterData>(
      `/api/twitter/data/${encodeURIComponent(username)}`
    );
  }

  async getTwitterUsers(): Promise<ApiResponse<TwitterData[]>> {
    return this.fetchApi<TwitterData[]>("/api/twitter/users");
  }

  // CSV Analysis APIs
  async uploadCSV(file: File): Promise<ApiResponse<CSVUploadResponse>> {
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${this.baseUrl}/api/csv/upload`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(
          errorData.message || `HTTP ${response.status}: ${response.statusText}`
        );
      }

      const data = await response.json();
      return {
        success: true,
        data,
      };
    } catch (error) {
      console.error("CSV Upload Error:", error);
      return {
        success: false,
        error:
          error instanceof Error ? error.message : "Unknown error occurred",
      };
    }
  }

  async analyzeCSV(
    request: CSVAnalysisRequest
  ): Promise<ApiResponse<AnalysisResult>> {
    // Handle direct text input by creating a simple CSV format
    let fileContent = request.file_content;
    let textColumn = request.text_column || "text";

    // Check if this is base64 encoded text (not CSV)
    try {
      const decoded = atob(fileContent);
      // If it doesn't contain comma/newlines, treat as simple text and wrap in CSV format
      if (!decoded.includes(",") && !decoded.includes("\n")) {
        const csvContent = `text\n"${decoded.replace(/"/g, '""')}"`;
        fileContent = btoa(csvContent);
        textColumn = "text";
      }
    } catch (e) {
      // If not base64, assume it's already proper format
    }

    return this.fetchApi<AnalysisResult>("/api/csv/analyze", {
      method: "POST",
      body: JSON.stringify({
        ...request,
        file_content: fileContent,
        text_column: textColumn,
      }),
    });
  }

  async analyzeSimpleText(
    request: SimpleTextAnalysisRequest
  ): Promise<ApiResponse<AnalysisResult>> {
    return this.fetchApi<AnalysisResult>("/api/csv/analyze-text", {
      method: "POST",
      body: JSON.stringify(request),
    });
  }

  async getCSVData(fileId: string): Promise<ApiResponse<CSVUploadResponse>> {
    return this.fetchApi<CSVUploadResponse>(`/api/csv/data/${fileId}`);
  }

  async getCSVFiles(): Promise<ApiResponse<CSVUploadResponse[]>> {
    return this.fetchApi<CSVUploadResponse[]>("/api/csv/files");
  }

  // Analysis APIs
  async getAnalysisResult(
    analysisId: string
  ): Promise<ApiResponse<AnalysisResult>> {
    return this.fetchApi<AnalysisResult>(`/api/analysis/results/${analysisId}`);
  }

  async getAnalysisHistory(): Promise<ApiResponse<AnalysisResult[]>> {
    return this.fetchApi<AnalysisResult[]>("/api/analysis/history");
  }

  async deleteAnalysis(
    analysisId: string
  ): Promise<ApiResponse<{ message: string }>> {
    return this.fetchApi<{ message: string }>(`/api/analysis/${analysisId}`, {
      method: "DELETE",
    });
  }

  // Model Status
  async getModelStatus(): Promise<ApiResponse<Record<string, any>>> {
    return this.fetchApi<Record<string, any>>("/api/analysis/models/status");
  }
}

// Create and export the API service instance
export const apiService = new ApiService();

// Export utility functions
export const formatApiError = (error: string | undefined): string => {
  if (!error) return "An unknown error occurred";
  return error.charAt(0).toUpperCase() + error.slice(1);
};

export const isApiError = (response: ApiResponse): boolean => {
  return !response.success || !!response.error;
};

export default apiService;
