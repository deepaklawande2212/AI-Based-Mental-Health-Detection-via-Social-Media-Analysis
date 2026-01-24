# Mental Health Detection System - Client Documentation

## 🚀 Overview

The client is a React-based frontend application that provides an intuitive interface for mental health analysis. It displays real-time results from multiple AI models with interactive charts and detailed metrics.

## 📁 Project Structure

```
client/
├── src/
│   ├── components/           # React components
│   │   ├── AnalysisReport.tsx    # Main analysis results display
│   │   ├── AnalysisProgressModal.tsx  # Progress tracking modal
│   │   ├── BackendStatus.tsx     # Server health indicator
│   │   ├── ErrorBoundary.tsx     # Error handling
│   │   ├── Footer.tsx           # Footer component
│   │   ├── LoadingSpinner.tsx   # Loading indicators
│   │   └── Navbar.tsx           # Navigation bar
│   ├── pages/               # Page components
│   │   ├── Dashboard.tsx        # Main dashboard
│   │   ├── Home.tsx            # Landing page
│   │   ├── About.tsx           # About page
│   │   └── ApiTest.tsx         # API testing page
│   ├── services/            # API services
│   │   └── api.ts              # API communication
│   ├── hooks/               # Custom React hooks
│   │   └── useApi.ts           # API integration hooks
│   ├── utils/               # Utility functions
│   │   ├── transforms.ts       # Data transformation
│   │   ├── helpers.ts          # Helper functions
│   │   └── api.ts             # API utilities
│   └── constants/           # Application constants
│       └── index.ts           # Global constants
├── public/                 # Static assets
├── package.json           # Dependencies
└── vite.config.ts         # Vite configuration
```

## 🛠️ Setup Instructions

### 1. Environment Setup

```bash
# Navigate to client directory
cd client

# Install dependencies
npm install

# Start development server
npm run dev
```

Client will start on: `http://localhost:3000`

### 2. Build for Production

```bash
# Build the application
npm run build

# Preview production build
npm run preview
```

## 🎯 Dashboard Features

### 1. Main Dashboard (`/dashboard`)

**Location**: `src/pages/Dashboard.tsx`

**Features**:

- **Text Input**: Direct text analysis
- **File Upload**: CSV file analysis with drag-and-drop
- **Real-time Progress**: Analysis progress modal
- **Model Status**: Individual model loading status
- **Results Display**: Comprehensive analysis results

### 2. Analysis Progress Modal

**Location**: `src/components/AnalysisProgressModal.tsx`

**Features**:

- **Progress Bar**: Real-time progress tracking
- **Step Indicators**: Current analysis step
- **Model Status**: Individual model processing status
- **Animated UI**: Smooth transitions and animations

### 3. Analysis Report Component

**Location**: `src/components/AnalysisReport.tsx`

**Features**:

- **Sentiment Analysis**: Pie chart with percentage breakdown
- **Emotion Detection**: Bar chart with emotion scores
- **Risk Assessment**: Risk level with confidence
- **Individual Model Results**: Detailed results from each model
- **Detailed Metrics**: Comprehensive analysis metrics
- **Recommendations**: Personalized recommendations

## 📊 Data Display Components

### 1. Sentiment Analysis Chart

- **Type**: Pie Chart (Recharts)
- **Data**: Positive, Negative, Neutral percentages
- **Features**:
  - Filters out zero values
  - Shows percentage labels
  - Color-coded segments

### 2. Emotion Detection Chart

- **Type**: Bar Chart (Recharts)
- **Data**: Joy, Sadness, Anger, Fear, Surprise, Disgust
- **Features**:
  - Converts decimal to percentage (0-100)
  - Filters valid data (0-100 range)
  - Responsive design

### 3. Risk Assessment Display

- **Type**: Text with confidence indicator
- **Data**: Low/Medium/High risk with confidence score
- **Features**:
  - Color-coded risk levels
  - Confidence percentage display
  - Risk factors list

### 4. Individual Model Results

- **Type**: Expandable sections
- **Data**: Results from each AI model
- **Features**:
  - Model name and status
  - Confidence scores
  - Sentiment breakdown
  - Risk assessment

## 🧪 Testing Procedures

### 1. Test Dashboard with Sample Data

#### Positive Text Sample:

```
I am feeling so happy and excited about life today! Everything seems wonderful and I can't wait to see what the future holds.
```

**Expected Results**:

- **Sentiment**: Mostly positive
- **Emotions**: High joy, low negative emotions
- **Risk Level**: Low
- **Confidence**: High (>80%)

#### Negative Text Sample:

```
I feel so hopeless and depressed. Everything seems meaningless and I want to give up. Life is not worth living anymore.
```

**Expected Results**:

- **Sentiment**: Mostly negative
- **Emotions**: High sadness, possible fear
- **Risk Level**: High
- **Confidence**: High (>80%)

#### High Risk Text Sample:

```
I feel like I want to end it all. I can't take this life anymore. I want to commit suicide.
```

**Expected Results**:

- **Sentiment**: Negative
- **Emotions**: High sadness, fear
- **Risk Level**: High (crisis)
- **Confidence**: High (>90%)

### 2. Test File Upload

1. **Create Test CSV**:

```csv
text,label
I am feeling so happy and excited about life today!,positive
I feel so hopeless and depressed.,negative
```

2. **Upload Process**:
   - Drag and drop CSV file
   - Select text column
   - Choose analysis type
   - Click "Start Analysis"

### 3. Test Progress Modal

1. **Start Analysis**: Upload text or file
2. **Observe Progress**:
   - Uploading (10%)
   - Preprocessing (20%)
   - Analyzing (40-90%)
   - Completing (100%)
3. **Model Status**: Watch individual model processing

## 📈 Data Flow

### 1. Input Processing

```
User Input → Text/File Validation → API Request → Backend Processing
```

### 2. API Communication

```typescript
// API Request Structure
{
  file_content: "base64_encoded_content",
  filename: "test.csv",
  analysis_type: "comprehensive",
  text_column: "text"
}
```

### 3. Response Processing

```typescript
// API Response Structure
{
  analysis_id: "uuid",
  model_results: [...],
  final_sentiment: {...},
  final_emotions: {...},
  final_risk: {...},
  best_model: "bert"
}
```

### 4. Data Transformation

```typescript
// Transform API response to frontend format
const transformedData = {
  sentiment: {...},
  emotions: {...},
  risk: {...},
  confidence: 85, // Converted to percentage
  modelResults: [...]
}
```

## 🎨 UI Components

### 1. Navigation

- **Navbar**: Main navigation with links
- **Breadcrumbs**: Current page indication
- **Status Indicator**: Backend connection status

### 2. Input Forms

- **Text Input**: Large textarea for direct input
- **File Upload**: Drag-and-drop zone with validation
- **Analysis Options**: Type selection and configuration

### 3. Results Display

- **Charts**: Interactive charts with tooltips
- **Metrics**: Detailed numerical data
- **Recommendations**: Actionable advice
- **Model Comparison**: Side-by-side model results

## 🔧 Technical Implementation

### 1. State Management

```typescript
// Dashboard State
const [analysisResult, setAnalysisResult] = useState(null);
const [showProgressModal, setShowProgressModal] = useState(false);
const [currentStep, setCurrentStep] = useState("");
const [progress, setProgress] = useState(0);
const [modelStatus, setModelStatus] = useState({});
```

### 2. API Integration

```typescript
// Custom Hook for API
const { analyzeCSV, analyzeTwitter, healthStatus } = useApi();

// Usage
const result = await analyzeCSV(fileData);
```

### 3. Data Transformation

```typescript
// Transform API response
const transformedData = transformAnalysisResult(apiResponse);

// Convert confidence to percentage
const confidencePercentage = Math.round(confidence * 100);
```

### 4. Error Handling

```typescript
// Error Boundary
<ErrorBoundary>
  <Dashboard />
</ErrorBoundary>;

// API Error Handling
try {
  const result = await analyzeCSV(data);
} catch (error) {
  toast.error(`Analysis failed: ${error.message}`);
}
```

## 📱 Responsive Design

### 1. Mobile Optimization

- **Touch-friendly**: Large buttons and inputs
- **Responsive charts**: Adapt to screen size
- **Collapsible sections**: Save space on small screens

### 2. Desktop Features

- **Full-width layout**: Utilize screen space
- **Side-by-side comparison**: Model results comparison
- **Advanced interactions**: Hover effects and tooltips

## 🚨 Troubleshooting

### Common Issues

1. **Backend Connection**: Check if server is running on port 8000
2. **CORS Errors**: Ensure backend CORS is configured
3. **File Upload Issues**: Check file format and size
4. **Chart Display**: Verify data transformation

### Debug Commands

```bash
# Check client status
curl http://localhost:3000

# Check backend connection
curl http://localhost:8000/api/health/health

# Test API directly
curl -X POST "http://localhost:8000/api/csv/analyze" \
  -H "Content-Type: application/json" \
  -d '{"file_content": "dGV4dCxsYWJlbApUZXN0IHRleHQsbmV1dHJhbA==", "filename": "test.csv", "analysis_type": "comprehensive", "text_column": "text"}'
```

## 📊 Performance Metrics

- **Initial Load Time**: ~2-3 seconds
- **Analysis Response Time**: ~3-5 seconds
- **Chart Rendering**: <1 second
- **File Upload**: Depends on file size
- **Memory Usage**: ~50-100MB

## 🔒 Security Considerations

- **Input Validation**: Client-side validation
- **File Type Checking**: CSV validation
- **Error Handling**: No sensitive data in errors
- **CORS**: Proper cross-origin configuration

## 🎯 Best Practices

### 1. User Experience

- **Loading States**: Always show loading indicators
- **Error Messages**: Clear, actionable error messages
- **Progress Feedback**: Real-time progress updates
- **Responsive Design**: Works on all devices

### 2. Data Handling

- **Validation**: Validate all inputs
- **Transformation**: Consistent data transformation
- **Caching**: Cache results when appropriate
- **Error Recovery**: Graceful error handling

### 3. Performance

- **Lazy Loading**: Load components on demand
- **Optimization**: Optimize bundle size
- **Caching**: Cache API responses
- **Debouncing**: Debounce user inputs
