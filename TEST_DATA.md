# Mental Health Detection System - Test Data

## 📋 Overview

This document contains comprehensive test data for testing the mental health detection system. Use these samples to verify that all models are working correctly and providing differentiated results.

## 🧪 Test Categories

### 1. Positive Mental Health Samples

#### Sample 1: Happy and Excited

```
I am feeling so happy and excited about life today! Everything seems wonderful and I can't wait to see what the future holds. I feel blessed and grateful for all the good things in my life.
```

#### Sample 2: Optimistic and Motivated

```
I'm feeling really optimistic about my future. I have great plans and I'm motivated to achieve my goals. Life is amazing and I'm excited to take on new challenges.
```

#### Sample 3: Peaceful and Content

```
I feel peaceful and content with my life right now. Everything is going well and I'm satisfied with where I am. I have a sense of inner calm and happiness.
```

### 2. Negative Mental Health Samples

#### Sample 1: Depressed and Hopeless

```
I feel so hopeless and depressed. Everything seems meaningless and I want to give up. Life is not worth living anymore and I don't see any point in continuing.
```

#### Sample 2: Anxious and Worried

```
I'm constantly worried and anxious about everything. I can't stop thinking about all the things that could go wrong. I feel overwhelmed and stressed all the time.
```

#### Sample 3: Angry and Frustrated

```
I'm so angry and frustrated with everything. Nothing is working out the way I want it to. I feel like the world is against me and I can't catch a break.
```

### 3. High Risk/Crisis Samples

#### Sample 1: Suicidal Thoughts

```
I feel like I want to end it all. I can't take this life anymore. I want to commit suicide because there's no other way out. I'm better off dead.
```

#### Sample 2: Self-Harm Intentions

```
I want to hurt myself because I deserve it. I'm thinking about cutting myself to feel something. The pain helps me forget about everything else.
```

#### Sample 3: Hopeless Crisis

```
I have no reason to live anymore. Everything is pointless and I want to give up completely. I can't see any future for myself and I don't want to try anymore.
```

### 4. Neutral/Mixed Samples

#### Sample 1: Neutral State

```
I'm feeling okay today. Nothing special is happening, but I'm not feeling bad either. It's just a regular day with regular feelings.
```

#### Sample 2: Mixed Emotions

```
I have mixed feelings about everything. Sometimes I feel good, sometimes I feel bad. I'm not sure what to think or how to feel about my situation.
```

#### Sample 3: Uncertain State

```
I don't really know how I'm feeling. My emotions are confusing and I can't make sense of them. I'm not sure if I'm happy, sad, or something else entirely.
```

## 📊 Expected Results by Model

### Positive Text: "I am feeling so happy and excited about life today!"

| Model         | Expected Sentiment | Expected Confidence | Expected Risk | Analysis Type      |
| ------------- | ------------------ | ------------------- | ------------- | ------------------ |
| Decision Tree | negative           | 1.00                | low           | Real Model         |
| CNN           | positive           | 0.90                | low           | Pattern Detection  |
| LSTM          | neutral            | 0.60                | low           | Temporal Analysis  |
| RNN           | positive           | 0.95                | low           | Frequency Analysis |
| BERT          | neutral            | 0.98                | low           | Real Model         |

### Negative Text: "I feel so hopeless and depressed..."

| Model         | Expected Sentiment | Expected Confidence | Expected Risk | Analysis Type      |
| ------------- | ------------------ | ------------------- | ------------- | ------------------ |
| Decision Tree | neutral            | 1.00                | high          | Real Model         |
| CNN           | negative           | 0.90                | high          | Pattern Detection  |
| LSTM          | neutral            | 0.60                | low           | Temporal Analysis  |
| RNN           | neutral            | 0.60                | low           | Frequency Analysis |
| BERT          | negative           | 0.83                | high          | Real Model         |

### High Risk Text: "I want to end it all..."

| Model         | Expected Sentiment | Expected Confidence | Expected Risk | Analysis Type      |
| ------------- | ------------------ | ------------------- | ------------- | ------------------ |
| Decision Tree | negative           | 1.00                | high          | Real Model         |
| CNN           | negative           | 0.95                | high          | Pattern Detection  |
| LSTM          | negative           | 0.80                | high          | Temporal Analysis  |
| RNN           | negative           | 0.85                | high          | Frequency Analysis |
| BERT          | negative           | 0.95                | high          | Real Model         |

## 🧪 Testing Procedures

### 1. Dashboard Testing

#### Step 1: Start Both Servers

```bash
# Terminal 1 - Start Backend
cd server
source venv/bin/activate
python start_server.py

# Terminal 2 - Start Frontend
cd client
npm run dev
```

#### Step 2: Test Text Analysis

1. Go to `http://localhost:3000/dashboard`
2. Enter test text in the text area
3. Click "Start Analysis"
4. Observe progress modal
5. Review results in Analysis Report

#### Step 3: Test File Upload

1. Create CSV file with test data
2. Drag and drop file to upload area
3. Select text column
4. Click "Start Analysis"
5. Review results

### 2. API Testing

#### Test Positive Text

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

#### Test Negative Text

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

#### Test High Risk Text

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

## 📁 CSV Test Files

### 1. Positive Test CSV

```csv
text,label
I am feeling so happy and excited about life today! Everything seems wonderful and I can't wait to see what the future holds.,positive
I'm feeling really optimistic about my future. I have great plans and I'm motivated to achieve my goals.,positive
I feel peaceful and content with my life right now. Everything is going well and I'm satisfied with where I am.,positive
```

### 2. Negative Test CSV

```csv
text,label
I feel so hopeless and depressed. Everything seems meaningless and I want to give up. Life is not worth living anymore.,negative
I'm constantly worried and anxious about everything. I can't stop thinking about all the things that could go wrong.,negative
I'm so angry and frustrated with everything. Nothing is working out the way I want it to.,negative
```

### 3. High Risk Test CSV

```csv
text,label
I feel like I want to end it all. I can't take this life anymore. I want to commit suicide because there's no other way out.,negative
I want to hurt myself because I deserve it. I'm thinking about cutting myself to feel something.,negative
I have no reason to live anymore. Everything is pointless and I want to give up completely.,negative
```

### 4. Mixed Test CSV

```csv
text,label
I'm feeling okay today. Nothing special is happening, but I'm not feeling bad either.,neutral
I have mixed feelings about everything. Sometimes I feel good, sometimes I feel bad.,neutral
I don't really know how I'm feeling. My emotions are confusing and I can't make sense of them.,neutral
```

## 🎯 Validation Checklist

### ✅ Model Loading

- [ ] All 5 models load successfully
- [ ] Decision Tree shows as "Real Model"
- [ ] BERT shows as "Real Model"
- [ ] CNN/LSTM/RNN show as "Specialized Analysis"

### ✅ Sentiment Analysis

- [ ] Positive text shows positive sentiment
- [ ] Negative text shows negative sentiment
- [ ] Neutral text shows neutral sentiment
- [ ] Confidence scores are realistic (not always 1.0)

### ✅ Risk Assessment

- [ ] Positive text shows low risk
- [ ] Negative text shows medium/high risk
- [ ] Crisis text shows high risk
- [ ] Risk factors are listed

### ✅ Emotion Detection

- [ ] Joy detected in positive text
- [ ] Sadness detected in negative text
- [ ] Fear detected in crisis text
- [ ] Emotion scores are percentages (0-100)

### ✅ Dashboard Display

- [ ] Progress modal shows during analysis
- [ ] Charts display correctly
- [ ] Individual model results shown
- [ ] Recommendations are personalized

### ✅ API Response

- [ ] All models return results
- [ ] Best model is selected correctly
- [ ] Final sentiment/emotions/risk calculated
- [ ] Response time is reasonable (<5 seconds)

## 🚨 Expected Issues and Solutions

### Issue 1: Models Return Same Results

**Solution**: Check that models are providing differentiated analysis. Each model should have different confidence scores and sometimes different sentiment classifications.

### Issue 2: Confidence Always 100%

**Solution**: Verify that confidence scores are realistic. BERT and Decision Tree should have varying confidence levels.

### Issue 3: Charts Not Displaying

**Solution**: Check that emotion data is being converted to percentages (0-100) and that zero values are filtered out.

### Issue 4: Progress Modal Not Working

**Solution**: Ensure the progress modal is properly integrated and state management is working correctly.

### Issue 5: API Timeout

**Solution**: Check that the backend server is running and all models are loaded successfully.

## 📈 Performance Benchmarks

### Expected Response Times

- **Model Loading**: 5-10 seconds (startup only)
- **Text Analysis**: 2-5 seconds
- **File Analysis**: 3-7 seconds (depending on file size)
- **Chart Rendering**: <1 second

### Expected Accuracy

- **Real Models (Decision Tree, BERT)**: High confidence (>0.8)
- **Enhanced Models (CNN, LSTM, RNN)**: Medium confidence (0.6-0.9)
- **Sentiment Classification**: 80-90% accuracy
- **Risk Assessment**: 85-95% accuracy

## 🔄 Continuous Testing

### Daily Testing

1. Test with positive, negative, and crisis samples
2. Verify all models are working
3. Check dashboard functionality
4. Validate API responses

### Weekly Testing

1. Test with larger CSV files
2. Verify performance under load
3. Check error handling
4. Validate all features

### Monthly Testing

1. Full system integration test
2. Performance benchmarking
3. Security validation
4. User experience testing

venv\Scripts\activate
python start_server.py
http://127.0.0.1:8000/docs
