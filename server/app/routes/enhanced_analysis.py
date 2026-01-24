"""
Enhanced Analysis API Routes

This module provides enhanced REST API endpoints for Twitter and text analysis
with comprehensive model accuracy reporting and sentiment analysis.

Endpoints:
- POST /twitter/analyze - Analyze Twitter user with all models
- POST /text/analyze - Analyze text with all models
- GET /models/accuracy - Get model accuracy statistics
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
import asyncio
from loguru import logger

from pydantic import BaseModel
from app.models.request_models import (
    TwitterAnalysisRequest, 
    TextAnalysisRequest,
    AnalysisResponse,
    ModelResultResponse,
    SentimentResponse,
    EmotionResponse,
    RiskResponse
)
from app.models.database_models import AnalysisResult
from app.config.database import get_analysis_collection, get_twitter_collection, get_text_analyses_collection, get_twitter_analyses_collection
from app.services.ai_models import ModelManager
from app.services.sentiment_analysis import SentimentAnalyzer
from app.services.emotion_detection import EmotionDetector
from app.services.risk_assessment import RiskAssessor

router = APIRouter()


class EnhancedAnalysisResponse(BaseModel):
    """Enhanced analysis response with detailed model information"""
    
    analysis_id: str
    analysis_type: str
    input_data: str  # Username or text
    model_results: List[Dict[str, Any]]
    best_model: Dict[str, Any]
    overall_sentiment: Dict[str, Any]
    overall_emotions: Dict[str, Any]
    overall_risk: Dict[str, Any]
    processing_time: float
    status: str
    created_at: datetime
    message: str


@router.post("/twitter/analyze", response_model=EnhancedAnalysisResponse)
async def analyze_twitter_user_enhanced(request: TwitterAnalysisRequest):
    """
    Enhanced Twitter user analysis with all 5 models accuracy and sentiment analysis
    
    Args:
        request: TwitterAnalysisRequest containing username
        
    Returns:
        EnhancedAnalysisResponse: Comprehensive analysis results
    """
    try:
        logger.info(f"🐦 Enhanced Twitter analysis request for: @{request.username}")
        
        # Get model manager from app state
        from app.main import app
        if not hasattr(app.state, 'model_manager'):
            raise HTTPException(
                status_code=503,
                detail="AI models not available"
            )
        
        model_manager = app.state.model_manager
        
        # Initialize analysis services
        sentiment_analyzer = SentimentAnalyzer()
        emotion_detector = EmotionDetector()
        risk_assessor = RiskAssessor()
        
        # Collect Twitter data (simulate for now - you can integrate real Twitter API)
        # For now, we'll use a sample text based on the username
        sample_text = f"Sample tweets from @{request.username}: This is a test tweet for analysis purposes. #mentalhealth #wellness"
        
        # Run all models on the Twitter data
        model_predictions = await model_manager.predict_all_models(sample_text)
        
        if not model_predictions:
            raise HTTPException(
                status_code=500,
                detail="No model predictions were successful"
            )
        
        # Process each model result with enhanced information
        enhanced_model_results = []
        total_accuracy = 0
        successful_models = 0
        
        for prediction in model_predictions:
            if prediction.status == "success":
                # Get sentiment analysis for this model's prediction
                sentiment_result = await sentiment_analyzer.analyze(sample_text)
                
                # Get emotion detection
                emotion_result = await emotion_detector.detect_emotions(sample_text)
                
                # Get risk assessment - use model_name as prediction identifier
                risk_result = await risk_assessor.assess_risk(sample_text, prediction.model_name)
                
                # Calculate model-specific accuracy (you can customize this based on your models)
                model_accuracy = prediction.confidence * 100
                total_accuracy += model_accuracy
                successful_models += 1
                
                enhanced_result = {
                    "model_name": prediction.model_name,
                    "prediction": prediction.model_name,
                    "confidence": prediction.confidence,
                    "accuracy": model_accuracy,
                    "processing_time": prediction.prediction_time,
                    "status": prediction.status,
                    "sentiment": {
                        "positive": sentiment_result.get("positive", 0.0),
                        "negative": sentiment_result.get("negative", 0.0),
                        "neutral": sentiment_result.get("neutral", 0.0),
                        "overall": sentiment_result.get("overall", "neutral"),
                        "confidence": sentiment_result.get("confidence", 0.0)
                    },
                    "emotions": {
                        "joy": emotion_result.get("joy", 0.0),
                        "sadness": emotion_result.get("sadness", 0.0),
                        "anger": emotion_result.get("anger", 0.0),
                        "fear": emotion_result.get("fear", 0.0),
                        "surprise": emotion_result.get("surprise", 0.0),
                        "disgust": emotion_result.get("disgust", 0.0),
                        "dominant_emotion": emotion_result.get("dominant_emotion", "neutral"),
                        "confidence": emotion_result.get("confidence", 0.0)
                    },
                    "risk": {
                        "level": risk_result.get("level", "low"),
                        "score": risk_result.get("score", 0.0),
                        "factors": risk_result.get("factors", []),
                        "recommendations": risk_result.get("recommendations", []),
                        "confidence": risk_result.get("confidence", 0.0)
                    }
                }
            else:
                enhanced_result = {
                    "model_name": prediction.model_name,
                    "prediction": None,
                    "confidence": 0.0,
                    "accuracy": 0.0,
                    "processing_time": 0.0,
                    "status": prediction.status,
                    "error_message": prediction.error_message,
                    "sentiment": None,
                    "emotions": None,
                    "risk": None
                }
            
            enhanced_model_results.append(enhanced_result)
        
        # Calculate best model using sophisticated multi-factor approach
        best_model = None
        if successful_models > 0:
            # Define model preferences (lower number = higher preference)
            model_preferences = {
                "bert": 1,      # BERT is most sophisticated
                "lstm": 2,      # LSTM is good for sequence data
                "cnn": 3,       # CNN is good for pattern recognition
                "decision_tree": 4,  # Decision tree is simpler
                "rnn": 5        # RNN is basic
            }
            
            # Calculate weighted scores for each model
            best_score = -1
            for result in enhanced_model_results:
                if result.get("status") == "success":
                    confidence = result.get("confidence", 0)
                    accuracy = result.get("accuracy", 0)
                    model_name = result.get("model_name", "unknown")
                    
                    # Penalize artificially high confidence (above 0.9)
                    if confidence > 0.9:
                        confidence = 0.9 - (confidence - 0.9) * 0.5
                    
                    # Get model preference (lower is better)
                    preference = model_preferences.get(model_name.lower(), 5)
                    
                    # Calculate weighted score: (confidence * 0.4 + accuracy * 0.4 + preference_weight * 0.2)
                    preference_weight = (6 - preference) / 5.0  # Convert to 0-1 scale
                    weighted_score = (confidence * 0.4) + ((accuracy / 100) * 0.4) + (preference_weight * 0.2)
                    
                    if weighted_score > best_score:
                        best_score = weighted_score
                        best_model = result
            
            # If no model selected, use the first successful model
            if not best_model:
                for result in enhanced_model_results:
                    if result.get("status") == "success":
                        best_model = result
                        break
        
        # Calculate overall metrics
        overall_accuracy = total_accuracy / successful_models if successful_models > 0 else 0
        
        # Aggregate overall sentiment, emotions, and risk from best model
        overall_sentiment = best_model.get("sentiment", {})
        overall_emotions = best_model.get("emotions", {})
        overall_risk = best_model.get("risk", {})
        
        # Create analysis ID
        analysis_id = str(uuid.uuid4())
        
        # Store in twitter_analyses collection
        try:
            twitter_analyses_collection = await get_twitter_analyses_collection()
            twitter_analysis_data = {
                "analysis_id": analysis_id,
                "username": request.username,  # Twitter username analyzed
                "tweets_analyzed": sample_text,  # Tweets that were analyzed
                "max_tweets_requested": request.max_tweets,
                "model_results": enhanced_model_results,
                "best_model": {
                    "model_name": best_model.get("model_name", "unknown"),
                    "prediction": best_model.get("prediction", "unknown"),
                    "accuracy": best_model.get("accuracy", 0),
                    "confidence": best_model.get("confidence", 0)
                },
                "overall_accuracy": overall_accuracy,
                "overall_sentiment": overall_sentiment,
                "overall_emotions": overall_emotions,
                "overall_risk": overall_risk,
                "successful_models": successful_models,
                "failed_models": [p.model_name for p in model_predictions if p.status == "failed"],
                "total_models": len(model_predictions),
                "processing_time": sum(r.get("processing_time", 0) for r in enhanced_model_results),
                "status": "completed",
                "created_at": datetime.utcnow()
            }
            await twitter_analyses_collection.insert_one(twitter_analysis_data)
            logger.info(f"✅ Stored Twitter analysis in twitter_analyses collection: {analysis_id}")
        except Exception as e:
            logger.warning(f"⚠️ Could not store Twitter analysis result: {str(e)}")
        
        return EnhancedAnalysisResponse(
            analysis_id=analysis_id,
            analysis_type="twitter_enhanced",
            input_data=f"@{request.username}",
            model_results=enhanced_model_results,
            best_model=best_model,
            overall_sentiment=overall_sentiment,
            overall_emotions=overall_emotions,
            overall_risk=overall_risk,
            processing_time=sum(r.get("processing_time", 0) for r in enhanced_model_results),
            status="completed",
            created_at=datetime.utcnow(),
            message=f"Twitter analysis completed for @{request.username} with {successful_models} successful models"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in enhanced Twitter analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/text/analyze", response_model=EnhancedAnalysisResponse)
async def analyze_text_enhanced(request: TextAnalysisRequest):
    """
    Enhanced text analysis with all 5 models accuracy and sentiment analysis
    
    Args:
        request: TextAnalysisRequest containing text to analyze
        
    Returns:
        EnhancedAnalysisResponse: Comprehensive analysis results
    """
    try:
        logger.info(f"📝 Enhanced text analysis request for: {request.text[:50]}...")
        
        if not request.text or request.text.strip() == "":
            raise HTTPException(
                status_code=400,
                detail="No text provided for analysis"
            )
        
        # Get model manager from app state
        from app.main import app
        if not hasattr(app.state, 'model_manager'):
            raise HTTPException(
                status_code=503,
                detail="AI models not available"
            )
        
        model_manager = app.state.model_manager
        
        # Initialize analysis services
        sentiment_analyzer = SentimentAnalyzer()
        emotion_detector = EmotionDetector()
        risk_assessor = RiskAssessor()
        
        # Run all models on the text
        model_predictions = await model_manager.predict_all_models(request.text.strip())
        
        if not model_predictions:
            raise HTTPException(
                status_code=500,
                detail="No model predictions were successful"
            )
        
        # Process each model result with enhanced information
        enhanced_model_results = []
        total_accuracy = 0
        successful_models = 0
        
        for prediction in model_predictions:
            if prediction.status == "success":
                # Get sentiment analysis for this model's prediction
                sentiment_result = await sentiment_analyzer.analyze(request.text)
                
                # Get emotion detection
                emotion_result = await emotion_detector.detect_emotions(request.text)
                
                # Get risk assessment
                risk_result = await risk_assessor.assess_risk(request.text, prediction.model_name)
                
                # Calculate model-specific accuracy
                model_accuracy = prediction.confidence * 100
                total_accuracy += model_accuracy
                successful_models += 1
                
                enhanced_result = {
                    "model_name": prediction.model_name,
                    "prediction": prediction.model_name,
                    "confidence": prediction.confidence,
                    "accuracy": model_accuracy,
                    "processing_time": prediction.prediction_time,
                    "status": prediction.status,
                    "sentiment": {
                        "positive": sentiment_result.get("positive", 0.0),
                        "negative": sentiment_result.get("negative", 0.0),
                        "neutral": sentiment_result.get("neutral", 0.0),
                        "overall": sentiment_result.get("overall", "neutral"),
                        "confidence": sentiment_result.get("confidence", 0.0)
                    },
                    "emotions": {
                        "joy": emotion_result.get("joy", 0.0),
                        "sadness": emotion_result.get("sadness", 0.0),
                        "anger": emotion_result.get("anger", 0.0),
                        "fear": emotion_result.get("fear", 0.0),
                        "surprise": emotion_result.get("surprise", 0.0),
                        "disgust": emotion_result.get("disgust", 0.0),
                        "dominant_emotion": emotion_result.get("dominant_emotion", "neutral"),
                        "confidence": emotion_result.get("confidence", 0.0)
                    },
                    "risk": {
                        "level": risk_result.get("level", "low"),
                        "score": risk_result.get("score", 0.0),
                        "factors": risk_result.get("factors", []),
                        "recommendations": risk_result.get("recommendations", []),
                        "confidence": risk_result.get("confidence", 0.0)
                    }
                }
            else:
                enhanced_result = {
                    "model_name": prediction.model_name,
                    "prediction": None,
                    "confidence": 0.0,
                    "accuracy": 0.0,
                    "processing_time": 0.0,
                    "status": prediction.status,
                    "error_message": prediction.error_message,
                    "sentiment": None,
                    "emotions": None,
                    "risk": None
                }
            
            enhanced_model_results.append(enhanced_result)
        
        # Calculate best model using sophisticated multi-factor approach
        best_model = None
        if successful_models > 0:
            # Define model preferences (lower number = higher preference)
            model_preferences = {
                "bert": 1,      # BERT is most sophisticated
                "lstm": 2,      # LSTM is good for sequence data
                "cnn": 3,       # CNN is good for pattern recognition
                "decision_tree": 4,  # Decision tree is simpler
                "rnn": 5        # RNN is basic
            }
            
            # Calculate weighted scores for each model
            best_score = -1
            for result in enhanced_model_results:
                if result.get("status") == "success":
                    confidence = result.get("confidence", 0)
                    accuracy = result.get("accuracy", 0)
                    model_name = result.get("model_name", "unknown")
                    
                    # Penalize artificially high confidence (above 0.9)
                    if confidence > 0.9:
                        confidence = 0.9 - (confidence - 0.9) * 0.5
                    
                    # Get model preference (lower is better)
                    preference = model_preferences.get(model_name.lower(), 5)
                    
                    # Calculate weighted score: (confidence * 0.4 + accuracy * 0.4 + preference_weight * 0.2)
                    preference_weight = (6 - preference) / 5.0  # Convert to 0-1 scale
                    weighted_score = (confidence * 0.4) + ((accuracy / 100) * 0.4) + (preference_weight * 0.2)
                    
                    if weighted_score > best_score:
                        best_score = weighted_score
                        best_model = result
            
            # If no model selected, use the first successful model
            if not best_model:
                for result in enhanced_model_results:
                    if result.get("status") == "success":
                        best_model = result
                        break
        
        # Calculate overall metrics
        overall_accuracy = total_accuracy / successful_models if successful_models > 0 else 0
        
        # Aggregate overall sentiment, emotions, and risk from best model
        overall_sentiment = best_model.get("sentiment", {})
        overall_emotions = best_model.get("emotions", {})
        overall_risk = best_model.get("risk", {})
        
        # Create analysis ID
        analysis_id = str(uuid.uuid4())
        
        # Store in text_analyses collection
        try:
            text_analyses_collection = await get_text_analyses_collection()
            text_analysis_data = {
                "analysis_id": analysis_id,
                "user_text": request.text,  # Full text entered by user
                "model_results": enhanced_model_results,
                "best_model": {
                    "model_name": best_model.get("model_name", "unknown"),
                    "prediction": best_model.get("prediction", "unknown"),
                    "accuracy": best_model.get("accuracy", 0),
                    "confidence": best_model.get("confidence", 0)
                },
                "overall_accuracy": overall_accuracy,
                "overall_sentiment": overall_sentiment,
                "overall_emotions": overall_emotions,
                "overall_risk": overall_risk,
                "successful_models": successful_models,
                "failed_models": [p.model_name for p in model_predictions if p.status == "failed"],
                "total_models": len(model_predictions),
                "processing_time": sum(r.get("processing_time", 0) for r in enhanced_model_results),
                "status": "completed",
                "created_at": datetime.utcnow()
            }
            await text_analyses_collection.insert_one(text_analysis_data)
            logger.info(f"✅ Stored text analysis in text_analyses collection: {analysis_id}")
        except Exception as e:
            logger.warning(f"⚠️ Could not store text analysis result: {str(e)}")
        
        return EnhancedAnalysisResponse(
            analysis_id=analysis_id,
            analysis_type="text_enhanced",
            input_data=request.text[:100] + "..." if len(request.text) > 100 else request.text,
            model_results=enhanced_model_results,
            best_model=best_model,
            overall_sentiment=overall_sentiment,
            overall_emotions=overall_emotions,
            overall_risk=overall_risk,
            processing_time=sum(r.get("processing_time", 0) for r in enhanced_model_results),
            status="completed",
            created_at=datetime.utcnow(),
            message=f"Text analysis completed with {successful_models} successful models"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in enhanced text analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/models/accuracy")
async def get_model_accuracy_stats():
    """
    Get accuracy statistics for all models
    
    Returns:
        Dict: Model accuracy statistics
    """
    try:
        # Get recent analysis results from database
        analysis_collection = await get_analysis_collection()
        
        # Get last 100 analysis results
        recent_analyses = await analysis_collection.find(
            {"analysis_type": {"$in": ["twitter_enhanced", "text_enhanced"]}},
            {"model_results": 1, "created_at": 1}
        ).sort("created_at", -1).limit(100).to_list(length=100)
        
        # Calculate accuracy statistics
        model_stats = {
            "decision_tree": {"total": 0, "successful": 0, "avg_accuracy": 0.0},
            "cnn": {"total": 0, "successful": 0, "avg_accuracy": 0.0},
            "lstm": {"total": 0, "successful": 0, "avg_accuracy": 0.0},
            "rnn": {"total": 0, "successful": 0, "avg_accuracy": 0.0},
            "bert": {"total": 0, "successful": 0, "avg_accuracy": 0.0}
        }
        
        for analysis in recent_analyses:
            for model_result in analysis.get("model_results", []):
                model_name = model_result.get("model_name", "").lower()
                if model_name in model_stats:
                    model_stats[model_name]["total"] += 1
                    if model_result.get("status") == "success":
                        model_stats[model_name]["successful"] += 1
                        model_stats[model_name]["avg_accuracy"] += model_result.get("accuracy", 0)
        
        # Calculate averages
        for model_name, stats in model_stats.items():
            if stats["successful"] > 0:
                stats["avg_accuracy"] = stats["avg_accuracy"] / stats["successful"]
            stats["success_rate"] = (stats["successful"] / stats["total"] * 100) if stats["total"] > 0 else 0
        
        return {
            "model_statistics": model_stats,
            "total_analyses": len(recent_analyses),
            "last_updated": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting model accuracy stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}") 


@router.get("/all-analyses")
async def get_all_analyses_with_accuracy(
    limit: int = 50,
    analysis_type: Optional[str] = None,
    sort_by: str = "created_at",
    sort_order: str = "desc"
):
    """
    Get all analysis results with accuracy data from the unified collection
    
    Args:
        limit: Number of results to return (default: 50)
        analysis_type: Filter by analysis type (twitter_enhanced, text_enhanced, csv)
        sort_by: Field to sort by (created_at, overall_accuracy, best_model_accuracy)
        sort_order: Sort order (asc, desc)
        
    Returns:
        List of analysis results with accuracy data
    """
    try:
        analysis_collection = await get_analysis_collection()
        
        # Build query
        query = {}
        if analysis_type:
            query["analysis_type"] = analysis_type
        
        # Build sort
        sort_direction = -1 if sort_order.lower() == "desc" else 1
        sort_criteria = [(sort_by, sort_direction)]
        
        # Get analyses
        analyses = await analysis_collection.find(query).sort(sort_criteria).limit(limit).to_list(length=limit)
        
        # Format results
        formatted_analyses = []
        for analysis in analyses:
            formatted_analysis = {
                "analysis_id": analysis.get("analysis_id"),
                "analysis_type": analysis.get("analysis_type"),
                "input_type": analysis.get("input_type", "unknown"),
                "input_data": analysis.get("input_data", analysis.get("text", analysis.get("username", "N/A"))),
                "created_at": analysis.get("created_at"),
                "status": analysis.get("status", "unknown"),
                "processing_time": analysis.get("processing_time", 0),
                "accuracy_summary": analysis.get("accuracy_summary", {}),
                "overall_accuracy": analysis.get("overall_accuracy", 0),
                "best_model": analysis.get("best_model", {}),
                "successful_models": analysis.get("successful_models", []),
                "failed_models": analysis.get("failed_models", []),
                "total_models": analysis.get("total_models", 0),
                "overall_sentiment": analysis.get("overall_sentiment", {}),
                "overall_emotions": analysis.get("overall_emotions", {}),
                "overall_risk": analysis.get("overall_risk", {})
            }
            formatted_analyses.append(formatted_analysis)
        
        return {
            "total_analyses": len(formatted_analyses),
            "analyses": formatted_analyses,
            "query": {
                "limit": limit,
                "analysis_type": analysis_type,
                "sort_by": sort_by,
                "sort_order": sort_order
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Error retrieving analyses: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/accuracy-stats")
async def get_comprehensive_accuracy_stats():
    """
    Get comprehensive accuracy statistics for all analysis types
    
    Returns:
        Comprehensive accuracy statistics
    """
    try:
        analysis_collection = await get_analysis_collection()
        
        # Get all analyses with accuracy data
        all_analyses = await analysis_collection.find({}).to_list(length=1000)
        
        # Initialize stats
        stats = {
            "total_analyses": len(all_analyses),
            "by_analysis_type": {},
            "by_model": {
                "decision_tree": {"total": 0, "successful": 0, "avg_accuracy": 0.0, "total_accuracy": 0.0},
                "cnn": {"total": 0, "successful": 0, "avg_accuracy": 0.0, "total_accuracy": 0.0},
                "lstm": {"total": 0, "successful": 0, "avg_accuracy": 0.0, "total_accuracy": 0.0},
                "rnn": {"total": 0, "successful": 0, "avg_accuracy": 0.0, "total_accuracy": 0.0},
                "bert": {"total": 0, "successful": 0, "avg_accuracy": 0.0, "total_accuracy": 0.0}
            },
            "overall_stats": {
                "avg_overall_accuracy": 0.0,
                "best_accuracy": 0.0,
                "worst_accuracy": 100.0,
                "most_used_model": "unknown",
                "most_successful_model": "unknown"
            }
        }
        
        total_overall_accuracy = 0
        model_usage = {}
        
        for analysis in all_analyses:
            analysis_type = analysis.get("analysis_type", "unknown")
            
            # Count by analysis type
            if analysis_type not in stats["by_analysis_type"]:
                stats["by_analysis_type"][analysis_type] = {
                    "count": 0,
                    "avg_accuracy": 0.0,
                    "total_accuracy": 0.0
                }
            stats["by_analysis_type"][analysis_type]["count"] += 1
            
            # Process accuracy summary
            accuracy_summary = analysis.get("accuracy_summary", {})
            overall_accuracy = analysis.get("overall_accuracy", 0)
            
            if overall_accuracy > 0:
                total_overall_accuracy += overall_accuracy
                stats["by_analysis_type"][analysis_type]["total_accuracy"] += overall_accuracy
                
                # Track best/worst accuracy
                if overall_accuracy > stats["overall_stats"]["best_accuracy"]:
                    stats["overall_stats"]["best_accuracy"] = overall_accuracy
                if overall_accuracy < stats["overall_stats"]["worst_accuracy"]:
                    stats["overall_stats"]["worst_accuracy"] = overall_accuracy
            
            # Process model results
            models_with_accuracy = accuracy_summary.get("models_with_accuracy", [])
            for model_data in models_with_accuracy:
                model_name = model_data.get("model_name", "").lower()
                accuracy = model_data.get("accuracy", 0)
                status = model_data.get("status", "unknown")
                
                if model_name in stats["by_model"]:
                    stats["by_model"][model_name]["total"] += 1
                    if status == "success":
                        stats["by_model"][model_name]["successful"] += 1
                        stats["by_model"][model_name]["total_accuracy"] += accuracy
                    
                    # Track model usage
                    model_usage[model_name] = model_usage.get(model_name, 0) + 1
        
        # Calculate averages
        if stats["total_analyses"] > 0:
            stats["overall_stats"]["avg_overall_accuracy"] = total_overall_accuracy / stats["total_analyses"]
        
        # Calculate model averages
        for model_name, model_stats in stats["by_model"].items():
            if model_stats["successful"] > 0:
                model_stats["avg_accuracy"] = model_stats["total_accuracy"] / model_stats["successful"]
            model_stats["success_rate"] = (model_stats["successful"] / model_stats["total"]) * 100 if model_stats["total"] > 0 else 0
        
        # Calculate analysis type averages
        for analysis_type, type_stats in stats["by_analysis_type"].items():
            if type_stats["count"] > 0:
                type_stats["avg_accuracy"] = type_stats["total_accuracy"] / type_stats["count"]
        
        # Find most used and most successful models
        if model_usage:
            stats["overall_stats"]["most_used_model"] = max(model_usage, key=model_usage.get)
        
        most_successful_model = max(
            stats["by_model"].items(),
            key=lambda x: x[1]["avg_accuracy"] if x[1]["successful"] > 0 else 0
        )
        stats["overall_stats"]["most_successful_model"] = most_successful_model[0]
        
        return stats
        
    except Exception as e:
        logger.error(f"❌ Error calculating accuracy stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}") 


@router.get("/text-analyses")
async def get_text_analyses(
    limit: int = 50,
    sort_by: str = "created_at",
    sort_order: str = "desc"
):
    """
    Get all text analyses from text_analyses collection
    
    Args:
        limit: Number of results to return (default: 50)
        sort_by: Field to sort by (created_at, overall_accuracy, best_model_accuracy)
        sort_order: Sort order (asc, desc)
        
    Returns:
        List of text analysis results
    """
    try:
        text_analyses_collection = await get_text_analyses_collection()
        
        # Build sort
        sort_direction = -1 if sort_order.lower() == "desc" else 1
        sort_criteria = [(sort_by, sort_direction)]
        
        # Get text analyses
        text_analyses = await text_analyses_collection.find({}).sort(sort_criteria).limit(limit).to_list(length=limit)
        
        # Format results
        formatted_analyses = []
        for analysis in text_analyses:
            formatted_analysis = {
                "analysis_id": analysis.get("analysis_id"),
                "user_text": analysis.get("user_text", ""),
                "best_model": analysis.get("best_model", {}),
                "overall_accuracy": analysis.get("overall_accuracy", 0),
                "overall_sentiment": analysis.get("overall_sentiment", {}),
                "overall_emotions": analysis.get("overall_emotions", {}),
                "overall_risk": analysis.get("overall_risk", {}),
                "successful_models": analysis.get("successful_models", []),
                "failed_models": analysis.get("failed_models", []),
                "total_models": analysis.get("total_models", 0),
                "processing_time": analysis.get("processing_time", 0),
                "status": analysis.get("status", "unknown"),
                "created_at": analysis.get("created_at"),
                "model_results": analysis.get("model_results", [])
            }
            formatted_analyses.append(formatted_analysis)
        
        return {
            "total_text_analyses": len(formatted_analyses),
            "text_analyses": formatted_analyses,
            "query": {
                "limit": limit,
                "sort_by": sort_by,
                "sort_order": sort_order
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Error retrieving text analyses: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/twitter-analyses")
async def get_twitter_analyses(
    limit: int = 50,
    username: Optional[str] = None,
    sort_by: str = "created_at",
    sort_order: str = "desc"
):
    """
    Get all Twitter analyses from twitter_analyses collection
    
    Args:
        limit: Number of results to return (default: 50)
        username: Filter by specific username
        sort_by: Field to sort by (created_at, overall_accuracy, best_model_accuracy)
        sort_order: Sort order (asc, desc)
        
    Returns:
        List of Twitter analysis results
    """
    try:
        twitter_analyses_collection = await get_twitter_analyses_collection()
        
        # Build query
        query = {}
        if username:
            query["username"] = username
        
        # Build sort
        sort_direction = -1 if sort_order.lower() == "desc" else 1
        sort_criteria = [(sort_by, sort_direction)]
        
        # Get Twitter analyses
        twitter_analyses = await twitter_analyses_collection.find(query).sort(sort_criteria).limit(limit).to_list(length=limit)
        
        # Format results
        formatted_analyses = []
        for analysis in twitter_analyses:
            formatted_analysis = {
                "analysis_id": analysis.get("analysis_id"),
                "username": analysis.get("username", ""),
                "tweets_analyzed": analysis.get("tweets_analyzed", ""),
                "max_tweets_requested": analysis.get("max_tweets_requested", 0),
                "best_model": analysis.get("best_model", {}),
                "overall_accuracy": analysis.get("overall_accuracy", 0),
                "overall_sentiment": analysis.get("overall_sentiment", {}),
                "overall_emotions": analysis.get("overall_emotions", {}),
                "overall_risk": analysis.get("overall_risk", {}),
                "successful_models": analysis.get("successful_models", []),
                "failed_models": analysis.get("failed_models", []),
                "total_models": analysis.get("total_models", 0),
                "processing_time": analysis.get("processing_time", 0),
                "status": analysis.get("status", "unknown"),
                "created_at": analysis.get("created_at"),
                "model_results": analysis.get("model_results", [])
            }
            formatted_analyses.append(formatted_analysis)
        
        return {
            "total_twitter_analyses": len(formatted_analyses),
            "twitter_analyses": formatted_analyses,
            "query": {
                "limit": limit,
                "username": username,
                "sort_by": sort_by,
                "sort_order": sort_order
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Error retrieving Twitter analyses: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/text-analyses/{analysis_id}")
async def get_text_analysis_by_id(analysis_id: str):
    """
    Get a specific text analysis by ID
    
    Args:
        analysis_id: Analysis identifier
        
    Returns:
        Text analysis result
    """
    try:
        text_analyses_collection = await get_text_analyses_collection()
        
        analysis = await text_analyses_collection.find_one({"analysis_id": analysis_id})
        
        if not analysis:
            raise HTTPException(
                status_code=404,
                detail=f"Text analysis with ID {analysis_id} not found"
            )
        
        return {
            "analysis_id": analysis.get("analysis_id"),
            "user_text": analysis.get("user_text", ""),
            "best_model": analysis.get("best_model", {}),
            "overall_accuracy": analysis.get("overall_accuracy", 0),
            "overall_sentiment": analysis.get("overall_sentiment", {}),
            "overall_emotions": analysis.get("overall_emotions", {}),
            "overall_risk": analysis.get("overall_risk", {}),
            "successful_models": analysis.get("successful_models", []),
            "failed_models": analysis.get("failed_models", []),
            "total_models": analysis.get("total_models", 0),
            "processing_time": analysis.get("processing_time", 0),
            "status": analysis.get("status", "unknown"),
            "created_at": analysis.get("created_at"),
            "model_results": analysis.get("model_results", [])
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error retrieving text analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/twitter-analyses/{analysis_id}")
async def get_twitter_analysis_by_id(analysis_id: str):
    """
    Get a specific Twitter analysis by ID
    
    Args:
        analysis_id: Analysis identifier
        
    Returns:
        Twitter analysis result
    """
    try:
        twitter_analyses_collection = await get_twitter_analyses_collection()
        
        analysis = await twitter_analyses_collection.find_one({"analysis_id": analysis_id})
        
        if not analysis:
            raise HTTPException(
                status_code=404,
                detail=f"Twitter analysis with ID {analysis_id} not found"
            )
        
        return {
            "analysis_id": analysis.get("analysis_id"),
            "username": analysis.get("username", ""),
            "tweets_analyzed": analysis.get("tweets_analyzed", ""),
            "max_tweets_requested": analysis.get("max_tweets_requested", 0),
            "best_model": analysis.get("best_model", {}),
            "overall_accuracy": analysis.get("overall_accuracy", 0),
            "overall_sentiment": analysis.get("overall_sentiment", {}),
            "overall_emotions": analysis.get("overall_emotions", {}),
            "overall_risk": analysis.get("overall_risk", {}),
            "successful_models": analysis.get("successful_models", []),
            "failed_models": analysis.get("failed_models", []),
            "total_models": analysis.get("total_models", 0),
            "processing_time": analysis.get("processing_time", 0),
            "status": analysis.get("status", "unknown"),
            "created_at": analysis.get("created_at"),
            "model_results": analysis.get("model_results", [])
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error retrieving Twitter analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}") 