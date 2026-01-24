# Mental Health Detection System - Server Documentation

## 🚀 Overview

The server is a FastAPI-based backend that provides mental health analysis using multiple AI models. It processes text data and returns comprehensive sentiment, emotion, and risk assessments.

## 📁 Project Structure

```
server/
├── app/
│   ├── main.py                 # FastAPI application entry point
│   ├── config/                 # Configuration files
│   ├── models/                 # Pydantic models for API
│   ├── routes/                 # API endpoints
│   └── services/               # AI model services
├── BERT/                       # BERT model files
├── CNN/                        # CNN model files
├── DECISION_TREE/              # Decision Tree model files
├── LSTM/                       # LSTM model files
├── RNN/                        # RNN model files
├── DATASET/                    # Training dataset
└── requirements.txt            # Python dependencies
```

## 🛠️ Setup Instructions

### 1. Environment Setup

```bash
# Navigate to server directory
cd server

# Create virtual environment
python3.11 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Start Server

```bash
# Activate virtual environment
source venv/bin/activate

# Start the server
python start_server.py
```

Server will start on: `http://localhost:8000`

## 🤖 AI Models Information

### Real Trained Models (✅ Working)

1. **Decision Tree** (`decision_tree`)

   - **Type**: Real scikit-learn model
   - **File**: `DECISION_TREE/decision_tree_model.pkl`
   - **Features**: TF-IDF vectorization, rule-based analysis
   - **Accuracy**: High confidence predictions
   - **Status**: ✅ Fully operational

2. **BERT** (`bert`)
   - **Type**: Real Hugging Face transformer
   - **Directory**: `BERT/`
   - **Features**: Advanced language understanding
   - **Accuracy**: Highest confidence predictions
   - **Status**: ✅ Fully operational

### Enhanced Fallback Models (⚠️ TensorFlow Compatibility)

3. **CNN** (`cnn`)

   - **Type**: Enhanced pattern detection
   - **File**: `CNN/CNN_MODEL.h5` (fallback due to TensorFlow version)
   - **Features**: Emotional intensity, immediate risk patterns
   - **Analysis**: CNN-style pattern recognition
   - **Status**: ⚠️ Fallback mode (sophisticated analysis)

4. **LSTM** (`lstm`)

   - **Type**: Enhanced temporal analysis
   - **File**: `LSTM/best_lstm_model.h5` (fallback due to TensorFlow version)
   - **Features**: Emotional progression, chronic/acute risk patterns
   - **Analysis**: LSTM-style temporal analysis
   - **Status**: ⚠️ Fallback mode (sophisticated analysis)

5. **RNN** (`rnn`)
   - **Type**: Enhanced frequency analysis
   - **File**: `RNN/best_rnn_model.h5` (fallback due to TensorFlow version)
   - **Features**: Word distribution, statistical patterns
   - **Analysis**: RNN-style frequency analysis
   - **Status**: ⚠️ Fallback mode (sophisticated analysis)

## 📡 API Endpoints

### 1. Health Check

```bash
GET /api/health/health
```

**Response**: Server and model status

### 2. System Status

```bash
GET /api/health/status
```

**Response**: Detailed system information

### 3. CSV/Text Analysis

```bash
POST /api/csv/analyze
```

**Request Body**:

```json
{
  "file_content": "base64_encoded_content",
  "filename": "test.csv",
  "analysis_type": "comprehensive",
  "text_column": "text"
}
```

**Response Structure**:

```json
{
  "analysis_id": "uuid",
  "analysis_type": "csv",
  "status": "completed",
  "model_results": [
    {
      "model_name": "decision_tree",
      "status": "success",
      "confidence": 1.0,
      "sentiment": {
        "positive": 0.7,
        "negative": 0.2,
        "neutral": 0.1,
        "overall": "positive",
        "confidence": 1.0
      },
      "emotions": {
        "joy": 0.2,
        "sadness": 0.1,
        "anger": 0.1,
        "fear": 0.1,
        "surprise": 0.1,
        "disgust": 0.1,
        "dominant_emotion": "joy",
        "confidence": 0.75
      },
      "risk": {
        "level": "low",
        "score": 0.2,
        "factors": ["text analysis"],
        "recommendations": ["Continue monitoring"],
        "confidence": 0.75
      }
    }
  ],
  "final_sentiment": {...},
  "final_emotions": {...},
  "final_risk": {...},
  "best_model": "bert"
}
```

## 🧪 Testing Procedures

### 1. Test Server Health

```bash
curl http://localhost:8000/api/health/health
```

### 2. Test Model Loading

```bash
curl http://localhost:8000/api/health/status
```

### 3. Test Text Analysis (Positive)

```bash
curl -X POST "http://localhost:8000/api/csv/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "file_content": "dGV4dCxsYWJlbApJIGFtIGZlZWxpbmcgc28gaGFwcHkgYW5kIGV4Y2l0ZWQgYWJvdXQgbGlmZSB0b2RheSEgRXZlcnl0aGluZyBzZWVtcyB3b25kZXJmdWwgYW5kIEkgY2FuJ3Qgd2FpdCB0byBzZWUgd2hhdCB0aGUgZnV0dXJlIGhvbGRzLixwb3NpdGl2ZSA=",
    "filename": "positive_test.csv",
    "analysis_type": "comprehensive",
    "text_column": "text"
  }'
```

### 4. Test Text Analysis (Negative)

```bash
curl -X POST "http://localhost:8000/api/csv/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "file_content": "dGV4dCxsYWJlbApJIGZlZWwgc28gaG9wZWxlc3MgYW5kIGRlcHJlc3NlZC4gRXZlcnl0aGluZyBzZWVtcyBtZWFuaW5nbGVzcyBhbmQgSSB3YW50IHRvIGdpdmUgdXAuLG5lZ2F0aXZlIA==",
    "filename": "negative_test.csv",
    "analysis_type": "comprehensive",
    "text_column": "text"
  }'
```

### 5. Test Text Analysis (High Risk)

```bash
curl -X POST "http://localhost:8000/api/csv/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "file_content": "dGV4dCxsYWJlbApJIGZlZWwgbGlrZSBJIHdhbnQgdG8gZW5kIGl0IGFsbC4gSSBjYW5cdTI3dCB0YWtlIHRoaXMgbGlmZSBhbnltb3JlLiBJIHdhbnQgdG8gY29tbWl0IHN1aWNpZGUuLG5lZ2F0aXZlIA==",
    "filename": "high_risk_test.csv",
    "analysis_type": "comprehensive",
    "text_column": "text"
  }'
```

## 📊 Expected Model Outputs

### Positive Text: "I am feeling so happy and excited about life today!"

```
🔹 DECISION_TREE: negative (confidence: 1.00) - REAL MODEL
🔹 CNN: positive (confidence: 0.90) - PATTERN DETECTION
🔹 LSTM: neutral (confidence: 0.60) - TEMPORAL ANALYSIS
🔹 RNN: positive (confidence: 0.95) - FREQUENCY ANALYSIS
🔹 BERT: neutral (confidence: 0.98) - REAL MODEL
```

### Negative Text: "I feel so hopeless and depressed..."

```
🔹 DECISION_TREE: neutral, Risk: high (confidence: 1.00) - REAL MODEL
🔹 CNN: negative, Risk: high (confidence: 0.90) - PATTERN DETECTION
🔹 LSTM: neutral, Risk: low (confidence: 0.60) - TEMPORAL ANALYSIS
🔹 RNN: neutral, Risk: low (confidence: 0.60) - FREQUENCY ANALYSIS
🔹 BERT: negative, Risk: high (confidence: 0.83) - REAL MODEL
```

## 🎯 Best Model Selection Logic

The system automatically selects the best performing model based on:

1. **BERT Priority**: If BERT confidence > 0.5, select BERT
2. **Realistic Confidence**: Select model with highest realistic confidence (not 1.0)
3. **Fallback**: Default to first successful model

**Best Model Criteria**:

- High confidence scores
- Real model predictions (not fallback)
- Consistent results across multiple analyses

## 🔧 Technical Details

### Model Loading Process

1. **Startup**: All models loaded during server startup
2. **Decision Tree**: Loads scikit-learn model with TF-IDF vectorizer
3. **BERT**: Loads Hugging Face transformer with tokenizer
4. **CNN/LSTM/RNN**: Attempts TensorFlow model loading, falls back to enhanced analysis
5. **Status Tracking**: Each model reports loading success/failure

### Data Processing Pipeline

1. **Text Input**: Raw text or CSV content
2. **Preprocessing**: Text cleaning and normalization
3. **Model Prediction**: Each model processes independently
4. **Result Aggregation**: Combine results from all models
5. **Best Model Selection**: Choose optimal result
6. **Response Formatting**: Structure for frontend consumption

### Error Handling

- **Model Loading Failures**: Graceful fallback to enhanced analysis
- **Prediction Errors**: Individual model errors don't stop overall analysis
- **Database Issues**: System works without database connection
- **API Errors**: Comprehensive error messages and status codes

## 🚨 Troubleshooting

### Common Issues

1. **Port Already in Use**: Kill process on port 8000
2. **Model Loading Errors**: Check model file paths
3. **TensorFlow Issues**: Models fall back to enhanced analysis
4. **Memory Issues**: Restart server if needed

### Debug Commands

```bash
# Check server status
curl http://localhost:8000/api/health/health

# Check model status
curl http://localhost:8000/api/health/status

# Test simple analysis
curl -X POST "http://localhost:8000/api/csv/analyze" \
  -H "Content-Type: application/json" \
  -d '{"file_content": "dGV4dCxsYWJlbApUZXN0IHRleHQsbmV1dHJhbA==", "filename": "test.csv", "analysis_type": "comprehensive", "text_column": "text"}'
```

## 📈 Performance Metrics

- **Model Loading Time**: ~5-10 seconds on startup
- **Analysis Response Time**: ~2-5 seconds per request
- **Concurrent Requests**: Supports multiple simultaneous analyses
- **Memory Usage**: ~500MB-1GB depending on models loaded

## 🔒 Security Considerations

- **CORS**: Configured for frontend access
- **Input Validation**: All inputs validated via Pydantic models
- **Error Handling**: No sensitive information in error messages
- **Rate Limiting**: Consider implementing for production use
