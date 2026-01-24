/**
 * API Test Page - For development testing
 * This page allows easy testing of all API endpoints
 */

import { useState } from "react";
import { motion } from "framer-motion";
import {
  Play,
  CheckCircle,
  XCircle,
  Loader2,
  Code,
  Database,
} from "lucide-react";
import { apiService } from "../services/api";
import toast from "react-hot-toast";

interface TestResult {
  endpoint: string;
  status: "idle" | "loading" | "success" | "error";
  response?: any;
  error?: string;
  duration?: number;
}

const ApiTestPage = () => {
  const [results, setResults] = useState<Record<string, TestResult>>({});
  const [twitterUsername, setTwitterUsername] = useState("");
  const [testText, setTestText] = useState(
    ""
  );

  const updateResult = (endpoint: string, update: Partial<TestResult>) => {
    setResults((prev) => ({
      ...prev,
      [endpoint]: { ...prev[endpoint], endpoint, ...update },
    }));
  };

  const testEndpoint = async (
    endpoint: string,
    apiCall: () => Promise<any>
  ) => {
    const startTime = Date.now();
    updateResult(endpoint, { status: "loading" });

    try {
      const response = await apiCall();
      const duration = Date.now() - startTime;

      updateResult(endpoint, {
        status: "success",
        response,
        duration,
      });

      toast.success(`${endpoint} test successful`);
    } catch (error) {
      const duration = Date.now() - startTime;
      updateResult(endpoint, {
        status: "error",
        error: error instanceof Error ? error.message : "Unknown error",
        duration,
      });

      toast.error(`${endpoint} test failed`);
    }
  };

  const tests = [
    {
      name: "Health Check",
      endpoint: "/api/health/health",
      test: () => testEndpoint("health", () => apiService.getHealth()),
    },
    {
      name: "System Status",
      endpoint: "/api/health/status",
      test: () =>
        testEndpoint("system-status", () => apiService.getSystemStatus()),
    },
    {
      name: "Model Status",
      endpoint: "/api/analysis/models/status",
      test: () => testEndpoint("models", () => apiService.getModelStatus()),
    },
    {
      name: "CSV Files List",
      endpoint: "/api/csv/files",
      test: () => testEndpoint("csv-files", () => apiService.getCSVFiles()),
    },
    {
      name: "Twitter Users List",
      endpoint: "/api/twitter/users",
      test: () =>
        testEndpoint("twitter-users", () => apiService.getTwitterUsers()),
    },
    {
      name: "Analysis History",
      endpoint: "/api/analysis/history",
      test: () =>
        testEndpoint("history", () => apiService.getAnalysisHistory()),
    },
    {
      name: "Twitter Analysis",
      endpoint: "/api/twitter/analyze",
      test: () =>
        testEndpoint("twitter", () =>
          apiService.analyzeTwitter({
            username: twitterUsername,
            max_tweets: 15,
            include_retweets: false,
          })
        ),
    },
    {
      name: "CSV Analysis (Text)",
      endpoint: "/api/csv/analyze",
      test: () =>
        testEndpoint("csv", () =>
          apiService.analyzeCSV({
            file_content: btoa(testText),
            filename: "test.txt",
            analysis_type: "comprehensive",
            text_column: "text",
          })
        ),
    },
  ];

  const getStatusIcon = (status: TestResult["status"]) => {
    switch (status) {
      case "loading":
        return <Loader2 className="h-4 w-4 animate-spin text-blue-500" />;
      case "success":
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case "error":
        return <XCircle className="h-4 w-4 text-red-500" />;
      default:
        return <Play className="h-4 w-4 text-gray-400" />;
    }
  };

  const getStatusColor = (status: TestResult["status"]) => {
    switch (status) {
      case "loading":
        return "border-blue-200 bg-blue-50";
      case "success":
        return "border-green-200 bg-green-50";
      case "error":
        return "border-red-200 bg-red-50";
      default:
        return "border-gray-200 bg-white";
    }
  };

  const runAllTests = async () => {
    for (const test of tests) {
      await test.test();
      // Small delay between tests
      await new Promise((resolve) => setTimeout(resolve, 500));
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="mb-8"
        >
          <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4 flex items-center">
            <Code className="h-8 w-8 text-primary-600 mr-3" />
            API Test Dashboard
          </h1>
          <p className="text-lg text-gray-600">
            Test all backend API endpoints to verify integration
          </p>
        </motion.div>

        {/* Test Configuration */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.1 }}
          className="card mb-8"
        >
          <h2 className="text-xl font-semibold text-gray-900 mb-4">
            Test Configuration
          </h2>
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Twitter Username (for Twitter API test)
              </label>
              <input
                type="text"
                value={twitterUsername}
                onChange={(e) => setTwitterUsername(e.target.value)}
                className="input-field"
                placeholder="username"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Test Text (for CSV Analysis test)
              </label>
              <textarea
                value={testText}
                onChange={(e) => setTestText(e.target.value)}
                className="input-field h-20 resize-none"
                placeholder="Enter text to analyze..."
              />
            </div>
          </div>
          <button onClick={runAllTests} className="btn-primary mt-6">
            Run All Tests
          </button>
        </motion.div>

        {/* Test Results */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="space-y-4"
        >
          <h2 className="text-xl font-semibold text-gray-900 flex items-center">
            <Database className="h-5 w-5 text-primary-600 mr-2" />
            API Endpoints
          </h2>

          {tests.map((test) => {
            const result =
              results[
                test.endpoint.split("/")[test.endpoint.split("/").length - 1]
              ];

            return (
              <div
                key={test.endpoint}
                className={`border rounded-lg p-6 transition-all ${getStatusColor(
                  result?.status || "idle"
                )}`}
              >
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900">
                      {test.name}
                    </h3>
                    <p className="text-sm text-gray-600 font-mono">
                      {test.endpoint}
                    </p>
                    {result?.duration && (
                      <p className="text-xs text-gray-500">
                        Response time: {result.duration}ms
                      </p>
                    )}
                  </div>
                  <div className="flex items-center space-x-2">
                    {getStatusIcon(result?.status || "idle")}
                    <button
                      onClick={test.test}
                      disabled={result?.status === "loading"}
                      className="btn-secondary"
                    >
                      Test
                    </button>
                  </div>
                </div>

                {result?.error && (
                  <div className="mt-4 p-3 bg-red-100 border border-red-200 rounded text-red-700 text-sm">
                    <strong>Error:</strong> {result.error}
                  </div>
                )}

                {result?.response && (
                  <div className="mt-4">
                    <details className="cursor-pointer">
                      <summary className="text-sm font-medium text-gray-700 mb-2">
                        View Response Data
                      </summary>
                      <pre className="bg-gray-100 p-3 rounded text-xs overflow-auto max-h-40 text-gray-800">
                        {JSON.stringify(result.response, null, 2)}
                      </pre>
                    </details>
                  </div>
                )}
              </div>
            );
          })}
        </motion.div>

        {/* Instructions */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.3 }}
          className="card mt-8"
        >
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Instructions
          </h3>
          <div className="space-y-2 text-sm text-gray-600">
            <p>
              • Make sure the backend server is running on{" "}
              <code className="bg-gray-100 px-1 rounded">
                http://localhost:8000
              </code>
            </p>
            <p>
              • The backend status indicator in the navbar should show
              "Connected"
            </p>
            <p>• Click "Test" next to each endpoint to test individually</p>
            <p>• Use "Run All Tests" to test all endpoints sequentially</p>
            <p>• Check the browser console for detailed error messages</p>
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default ApiTestPage;
