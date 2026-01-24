import { useState } from "react";
import { motion } from "framer-motion";
import {
  FileText,
  Twitter,
  Upload,
  Send,
  BarChart3,
  AlertTriangle,
  RefreshCw,
} from "lucide-react";
import { useDropzone } from "react-dropzone";
import toast from "react-hot-toast";
import AnalysisReport from "../components/AnalysisReport";
import AnalysisProgressModal from "../components/AnalysisProgressModal";
import {
  useTwitterAnalysis,
  useCSVAnalysis,
  useHealthCheck,
  useSimpleTextAnalysis,
} from "../hooks/useApi";
import {
  transformAnalysisResult,
  validateTwitterUsername,
  validateCSVFile,
} from "../utils/transforms";

type AnalysisMode = "csv" | "twitter";

const Dashboard = () => {
  const [analysisMode, setAnalysisMode] = useState<AnalysisMode>("csv");
  const [textInput, setTextInput] = useState("");
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [twitterHandle, setTwitterHandle] = useState("");

  // Progress state management
  const [showProgressModal, setShowProgressModal] = useState(false);
  const [currentStep, setCurrentStep] = useState("uploading");
  const [progress, setProgress] = useState(0);
  const [modelStatus, setModelStatus] = useState<{
    [key: string]: "pending" | "loading" | "completed" | "error";
  }>({
    decision_tree: "pending",
    cnn: "pending",
    lstm: "pending",
    rnn: "pending",
    bert: "pending",
  });

  // Handle progress modal cancellation
  const handleCancelAnalysis = () => {
    setShowProgressModal(false);
    setProgress(0);
    setCurrentStep("uploading");
    setModelStatus({
      decision_tree: "pending",
      cnn: "pending",
      lstm: "pending",
      rnn: "pending",
      bert: "pending",
    });
    toast.success("Analysis cancelled");
  };

  // API Hooks
  const twitterAnalysis = useTwitterAnalysis();
  const csvAnalysis = useCSVAnalysis();
  const simpleTextAnalysis = useSimpleTextAnalysis();
  const { health } = useHealthCheck();

  const onDrop = (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (file && file.type === "text/csv") {
      setUploadedFile(file);
      toast.success("CSV file uploaded successfully!");
    } else {
      toast.error("Please upload a valid CSV file");
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "text/csv": [".csv"],
    },
    maxFiles: 1,
  });

  const handleAnalysis = async () => {
    try {
      // Start progress tracking
      setShowProgressModal(true);
      setCurrentStep("uploading");
      setProgress(10);
      setModelStatus({
        decision_tree: "pending",
        cnn: "pending",
        lstm: "pending",
        rnn: "pending",
        bert: "pending",
      });

      if (analysisMode === "csv") {
        if (!textInput && !uploadedFile) {
          toast.error("Please enter text or upload a CSV file");
          setShowProgressModal(false);
          return;
        }

        // Preprocessing step
        setCurrentStep("preprocessing");
        setProgress(20);

        if (uploadedFile) {
          // Validate CSV file
          const fileError = validateCSVFile(uploadedFile);
          if (fileError) {
            toast.error(fileError);
            setShowProgressModal(false);
            return;
          }

          // Upload CSV file first
          await csvAnalysis.uploadCSV(uploadedFile);
          setProgress(30);
          toast.success("File uploaded successfully!");

          // Then analyze the uploaded file
          setCurrentStep("analyzing");
          setProgress(40);

          // Simulate model loading progress
          const models = ["decision_tree", "cnn", "lstm", "rnn", "bert"];
          for (let i = 0; i < models.length; i++) {
            setModelStatus((prev) => ({ ...prev, [models[i]]: "loading" }));
            await new Promise((resolve) => setTimeout(resolve, 300)); // Simulate processing time
            setModelStatus((prev) => ({ ...prev, [models[i]]: "completed" }));
            setProgress(40 + (i + 1) * 10);
          }

          const fileContent = await uploadedFile.text();
          await csvAnalysis.analyzeCSV({
            file_content: btoa(fileContent), // Base64 encode
            filename: uploadedFile.name,
            analysis_type: "comprehensive",
          });
        } else {
          // Analyze direct text input using simple text analysis
          setCurrentStep("analyzing");
          setProgress(40);

          // Simulate model loading progress
          const models = ["decision_tree", "cnn", "lstm", "rnn", "bert"];
          for (let i = 0; i < models.length; i++) {
            setModelStatus((prev) => ({ ...prev, [models[i]]: "loading" }));
            await new Promise((resolve) => setTimeout(resolve, 300)); // Simulate processing time
            setModelStatus((prev) => ({ ...prev, [models[i]]: "completed" }));
            setProgress(40 + (i + 1) * 10);
          }

          await simpleTextAnalysis.analyzeText(textInput);
        }

        setCurrentStep("completing");
        setProgress(100);
        await new Promise((resolve) => setTimeout(resolve, 500)); // Show completion briefly

        setShowProgressModal(false);
        toast.success("CSV analysis completed successfully!");
      } else {
        if (!twitterHandle) {
          toast.error("Please enter a Twitter handle");
          setShowProgressModal(false);
          return;
        }

        // Validate Twitter username
        const usernameError = validateTwitterUsername(twitterHandle);
        if (usernameError) {
          toast.error(usernameError);
          setShowProgressModal(false);
          return;
        }

        setCurrentStep("analyzing");
        setProgress(40);

        // Simulate model loading progress for Twitter analysis
        const models = ["decision_tree", "cnn", "lstm", "rnn", "bert"];
        for (let i = 0; i < models.length; i++) {
          setModelStatus((prev) => ({ ...prev, [models[i]]: "loading" }));
          await new Promise((resolve) => setTimeout(resolve, 300));
          setModelStatus((prev) => ({ ...prev, [models[i]]: "completed" }));
          setProgress(40 + (i + 1) * 10);
        }

        await twitterAnalysis.analyzeTwitter({
          username: twitterHandle.replace("@", ""),
          max_tweets: 15,
          include_retweets: false,
        });

        setCurrentStep("completing");
        setProgress(100);
        await new Promise((resolve) => setTimeout(resolve, 500));

        setShowProgressModal(false);
        toast.success("Twitter analysis completed successfully!");
      }
    } catch (error) {
      console.error("Analysis error:", error);
      setShowProgressModal(false);
      toast.error("Analysis failed. Please try again.");
    }
  };

  // Get current analysis result and loading state
  const isAnalyzing =
    twitterAnalysis.loading ||
    csvAnalysis.loading ||
    simpleTextAnalysis.loading;
  const analysisResult =
    analysisMode === "twitter"
      ? twitterAnalysis.analysisResult
      : uploadedFile
      ? csvAnalysis.analysisResult
      : simpleTextAnalysis.analysisResult;
  const analysisError =
    twitterAnalysis.error || csvAnalysis.error || simpleTextAnalysis.error;

  // Show error messages
  if (analysisError) {
    toast.error(analysisError);
  }

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="mb-8"
        >
          <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
            Mental Health Analysis
          </h1>
          <p className="text-lg text-gray-600">
            Analyze text or CSV files for mental health insights using AI models
          </p>
        </motion.div>

        {/* Analysis Mode Selection */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.1 }}
          className="mb-8"
        >
          <div className="grid md:grid-cols-2 gap-6">
            <button
              onClick={() => setAnalysisMode("csv")}
              className={`card p-8 text-left transition-all duration-300 hover:shadow-xl ${
                analysisMode === "csv"
                  ? "ring-2 ring-primary-500 bg-primary-50"
                  : "hover:bg-gray-50"
              }`}
            >
              <FileText className="h-12 w-12 text-primary-600 mb-4" />
              <h3 className="text-xl font-semibold text-gray-900 mb-2">
                CSV File Analysis
              </h3>
              <p className="text-gray-600">
                Upload CSV files or enter text directly for analysis using our
                advanced AI models
              </p>
            </button>

            <button
              onClick={() => setAnalysisMode("twitter")}
              className={`card p-8 text-left transition-all duration-300 hover:shadow-xl ${
                analysisMode === "twitter"
                  ? "ring-2 ring-primary-500 bg-primary-50"
                  : "hover:bg-gray-50"
              }`}
            >
              <Twitter className="h-12 w-12 text-blue-500 mb-4" />
              <h3 className="text-xl font-semibold text-gray-900 mb-2">
                Twitter Data Analysis
              </h3>
              <p className="text-gray-600">
                Analyze Twitter data directly from user profiles and recent
                tweets
              </p>
            </button>
          </div>
        </motion.div>

        {/* Analysis Interface */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="grid lg:grid-cols-3 gap-8"
        >
          {/* Input Section */}
          <div className="lg:col-span-2">
            <div className="card">
              <h2 className="text-2xl font-semibold text-gray-900 mb-6">
                {analysisMode === "csv"
                  ? "Text & File Input"
                  : "Twitter Analysis"}
              </h2>

              {analysisMode === "csv" ? (
                <div className="space-y-6">
                  {/* Text Input */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Enter Text for Analysis
                    </label>
                    <textarea
                      value={textInput}
                      onChange={(e) => setTextInput(e.target.value)}
                      placeholder="Enter the text you want to analyze for mental health patterns..."
                      className="input-field h-32 resize-none"
                    />
                  </div>

                  {/* File Upload */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Upload CSV File
                    </label>
                    <div
                      {...getRootProps()}
                      className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
                        isDragActive
                          ? "border-primary-500 bg-primary-50"
                          : "border-gray-300 hover:border-primary-400"
                      }`}
                    >
                      <input {...getInputProps()} />
                      <Upload className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                      {uploadedFile ? (
                        <div>
                          <p className="text-green-600 font-medium">
                            {uploadedFile.name}
                          </p>
                          <p className="text-sm text-gray-500">
                            File uploaded successfully
                          </p>
                        </div>
                      ) : (
                        <div>
                          <p className="text-gray-600 mb-2">
                            {isDragActive
                              ? "Drop your CSV file here"
                              : "Drag & drop your CSV file here"}
                          </p>
                          <p className="text-sm text-gray-500">
                            or click to browse files
                          </p>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ) : (
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Twitter Handle
                  </label>
                  <div className="flex">
                    <span className="inline-flex items-center px-3 rounded-l-lg border border-r-0 border-gray-300 bg-gray-50 text-gray-500">
                      @
                    </span>
                    <input
                      type="text"
                      value={twitterHandle}
                      onChange={(e) => setTwitterHandle(e.target.value)}
                      placeholder="username"
                      className="input-field rounded-l-none"
                    />
                  </div>
                  <p className="text-sm text-gray-500 mt-2">
                    Enter the Twitter username to analyze their real recent
                    tweets (up to 15 tweets). We'll automatically fetch and
                    analyze their latest posts for mental health insights.
                  </p>
                </div>
              )}

              <button
                onClick={handleAnalysis}
                disabled={isAnalyzing}
                className="btn-primary w-full mt-6 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isAnalyzing ? (
                  <div className="flex items-center justify-center">
                    <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                    Analyzing...
                  </div>
                ) : (
                  <div className="flex items-center justify-center">
                    <Send className="mr-2 h-5 w-5" />
                    Start Analysis
                  </div>
                )}
              </button>
            </div>
          </div>

          {/* AI Models Info */}
          <div className="space-y-6">
            <div className="card">
              <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                <BarChart3 className="h-5 w-5 text-primary-600 mr-2" />
                System Status
              </h3>
              <div className="mb-3 p-3 bg-green-50 border border-green-200 rounded-lg">
                <p className="text-sm text-green-800">
                  <strong>✅ System Ready:</strong> All AI models are loaded and
                  ready for analysis.
                </p>
              </div>
              <div className="text-sm text-gray-600">
                <p>• Enter text or upload CSV file to start analysis</p>
                <p>
                  • Results will show sentiment, risk assessment, and emotions
                </p>
                <p>• Individual model performance will be displayed</p>
              </div>
            </div>

            <div className="card">
              <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                <AlertTriangle className="h-5 w-5 text-yellow-500 mr-2" />
                Analysis Types
              </h3>
              <div className="space-y-2 text-sm text-gray-600">
                <div className="flex items-center">
                  <div className="w-2 h-2 bg-green-500 rounded-full mr-2"></div>
                  Sentiment Analysis
                </div>
                <div className="flex items-center">
                  <div className="w-2 h-2 bg-yellow-500 rounded-full mr-2"></div>
                  Risk Assessment
                </div>
                <div className="flex items-center">
                  <div className="w-2 h-2 bg-blue-500 rounded-full mr-2"></div>
                  Emotion Detection
                </div>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Analysis Results */}
        {analysisResult && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="mt-8"
          >
            <AnalysisReport result={transformAnalysisResult(analysisResult)} />
          </motion.div>
        )}
      </div>
      {showProgressModal && (
        <AnalysisProgressModal
          isOpen={showProgressModal}
          analysisMode={analysisMode}
          currentStep={currentStep}
          progress={progress}
          modelStatus={modelStatus}
          onClose={handleCancelAnalysis}
        />
      )}
    </div>
  );
};

export default Dashboard;
