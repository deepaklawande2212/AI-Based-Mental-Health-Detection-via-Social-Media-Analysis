"""
Analysis API Routes

This module provides REST API endpoints for analysis results and history.

Endpoints:
- GET /results/{analysis_id} - Get specific analysis result
- GET /history - Get analysis history
- GET /models/status - Get AI model status
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List
from datetime import datetime, timedelta
from loguru import logger

from app.models.request_models import AnalysisResponse, ModelStatusResponse
from app.models.database_models import AnalysisResult
from app.config.database import get_analysis_collection


router = APIRouter()


@router.get("/results/{analysis_id}", response_model=AnalysisResponse)
async def get_analysis_result(analysis_id: str):
    """
    Get a specific analysis result by ID
    
    Args:
        analysis_id: Analysis identifier
        
    Returns:
        AnalysisResponse: Analysis result
    """
    try:
        logger.info(f"📊 API request to get analysis result: {analysis_id}")
        
        collection = await get_analysis_collection()
        doc = await collection.find_one({"analysis_id": analysis_id})
        
        if not doc:
            raise HTTPException(
                status_code=404,
                detail=f"Analysis result with ID {analysis_id} not found"
            )
        
        analysis_result = AnalysisResult(**doc)
        
        # Convert to response format
        from app.models.request_models import ModelResultResponse, SentimentResponse, EmotionResponse, RiskResponse
        
        model_results = []
        for pred in analysis_result.model_predictions:
            sentiment_resp = None
            if pred.sentiment:
                sentiment_resp = SentimentResponse(
                    positive=pred.sentiment.positive,
                    negative=pred.sentiment.negative,
                    neutral=pred.sentiment.neutral,
                    overall=pred.sentiment.overall,
                    confidence=pred.sentiment.confidence
                )
            
            emotion_resp = None
            if pred.emotions:
                emotion_resp = EmotionResponse(
                    joy=pred.emotions.joy,
                    sadness=pred.emotions.sadness,
                    anger=pred.emotions.anger,
                    fear=pred.emotions.fear,
                    surprise=pred.emotions.surprise,
                    disgust=pred.emotions.disgust,
                    dominant_emotion=pred.emotions.dominant_emotion,
                    confidence=pred.emotions.confidence
                )
            
            risk_resp = None
            if pred.risk:
                risk_resp = RiskResponse(
                    level=pred.risk.level,
                    score=pred.risk.score,
                    factors=pred.risk.factors,
                    recommendations=pred.risk.recommendations,
                    confidence=pred.risk.confidence
                )
            
            model_results.append(ModelResultResponse(
                model_name=pred.model_name,
                accuracy=pred.accuracy,
                processing_time=pred.prediction_time,
                status=pred.status,
                confidence=pred.confidence,
                sentiment=sentiment_resp,
                emotions=emotion_resp,
                risk=risk_resp,
                error_message=pred.error_message
            ))
        
        # Create final responses
        final_sentiment = None
        if analysis_result.final_sentiment:
            final_sentiment = SentimentResponse(
                positive=analysis_result.final_sentiment.positive,
                negative=analysis_result.final_sentiment.negative,
                neutral=analysis_result.final_sentiment.neutral,
                overall=analysis_result.final_sentiment.overall,
                confidence=analysis_result.final_sentiment.confidence
            )
        
        final_emotions = None
        if analysis_result.final_emotions:
            final_emotions = EmotionResponse(
                joy=analysis_result.final_emotions.joy,
                sadness=analysis_result.final_emotions.sadness,
                anger=analysis_result.final_emotions.anger,
                fear=analysis_result.final_emotions.fear,
                surprise=analysis_result.final_emotions.surprise,
                disgust=analysis_result.final_emotions.disgust,
                dominant_emotion=analysis_result.final_emotions.dominant_emotion,
                confidence=analysis_result.final_emotions.confidence
            )
        
        final_risk = None
        if analysis_result.final_risk:
            final_risk = RiskResponse(
                level=analysis_result.final_risk.level,
                score=analysis_result.final_risk.score,
                factors=analysis_result.final_risk.factors,
                recommendations=analysis_result.final_risk.recommendations,
                confidence=analysis_result.final_risk.confidence
            )
        
        # Calculate overall confidence
        successful_predictions = [p for p in analysis_result.model_predictions if p.status == "success"]
        avg_confidence = (
            sum(p.confidence for p in successful_predictions) / len(successful_predictions)
            if successful_predictions else 0.0
        )
        
        return AnalysisResponse(
            analysis_id=analysis_result.analysis_id,
            analysis_type=analysis_result.analysis_type,
            status=analysis_result.status,
            model_results=model_results,
            final_sentiment=final_sentiment,
            final_emotions=final_emotions,
            final_risk=final_risk,
            best_model=analysis_result.best_performing_model,
            successful_models=analysis_result.successful_models,
            failed_models=analysis_result.failed_models,
            processing_time=analysis_result.processing_time,
            created_at=analysis_result.created_at,
            confidence=avg_confidence
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting analysis result: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/history", response_model=List[AnalysisResponse])
async def get_analysis_history(
    analysis_type: Optional[str] = Query(None, description="Filter by analysis type (twitter/csv)"),
    limit: int = Query(50, description="Maximum number of results to return"),
    skip: int = Query(0, description="Number of results to skip"),
    days: Optional[int] = Query(None, description="Filter results from last N days")
):
    """
    Get analysis history with optional filters
    
    Args:
        analysis_type: Filter by analysis type
        limit: Maximum number of results
        skip: Number of results to skip
        days: Filter results from last N days
        
    Returns:
        List[AnalysisResponse]: List of analysis results
    """
    try:
        logger.info(f"📚 API request to get analysis history (type: {analysis_type}, limit: {limit})")
        
        collection = await get_analysis_collection()
        
        # Build query filter
        query_filter = {}
        
        if analysis_type:
            if analysis_type not in ["twitter", "csv"]:
                raise HTTPException(
                    status_code=400,
                    detail="analysis_type must be 'twitter' or 'csv'"
                )
            query_filter["analysis_type"] = analysis_type
        
        if days:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            query_filter["created_at"] = {"$gte": cutoff_date}
        
        # Execute query
        cursor = collection.find(query_filter).sort("created_at", -1).skip(skip).limit(limit)
        docs = await cursor.to_list(length=limit)
        
        # Convert to response format
        from app.models.request_models import ModelResultResponse, SentimentResponse, EmotionResponse, RiskResponse
        
        history = []
        for doc in docs:
            analysis_result = AnalysisResult(**doc)
            
            model_results = []
            for pred in analysis_result.model_predictions:
                sentiment_resp = None
                if pred.sentiment:
                    sentiment_resp = SentimentResponse(
                        positive=pred.sentiment.positive,
                        negative=pred.sentiment.negative,
                        neutral=pred.sentiment.neutral,
                        overall=pred.sentiment.overall,
                        confidence=pred.sentiment.confidence
                    )
                
                emotion_resp = None
                if pred.emotions:
                    emotion_resp = EmotionResponse(
                        joy=pred.emotions.joy,
                        sadness=pred.emotions.sadness,
                        anger=pred.emotions.anger,
                        fear=pred.emotions.fear,
                        surprise=pred.emotions.surprise,
                        disgust=pred.emotions.disgust,
                        dominant_emotion=pred.emotions.dominant_emotion,
                        confidence=pred.emotions.confidence
                    )
                
                risk_resp = None
                if pred.risk:
                    risk_resp = RiskResponse(
                        level=pred.risk.level,
                        score=pred.risk.score,
                        factors=pred.risk.factors,
                        recommendations=pred.risk.recommendations,
                        confidence=pred.risk.confidence
                    )
                
                model_results.append(ModelResultResponse(
                    model_name=pred.model_name,
                    accuracy=pred.accuracy,
                    processing_time=pred.prediction_time,
                    status=pred.status,
                    confidence=pred.confidence,
                    sentiment=sentiment_resp,
                    emotions=emotion_resp,
                    risk=risk_resp,
                    error_message=pred.error_message
                ))
            
            # Create final responses
            final_sentiment = None
            if analysis_result.final_sentiment:
                final_sentiment = SentimentResponse(
                    positive=analysis_result.final_sentiment.positive,
                    negative=analysis_result.final_sentiment.negative,
                    neutral=analysis_result.final_sentiment.neutral,
                    overall=analysis_result.final_sentiment.overall,
                    confidence=analysis_result.final_sentiment.confidence
                )
            
            final_emotions = None
            if analysis_result.final_emotions:
                final_emotions = EmotionResponse(
                    joy=analysis_result.final_emotions.joy,
                    sadness=analysis_result.final_emotions.sadness,
                    anger=analysis_result.final_emotions.anger,
                    fear=analysis_result.final_emotions.fear,
                    surprise=analysis_result.final_emotions.surprise,
                    disgust=analysis_result.final_emotions.disgust,
                    dominant_emotion=analysis_result.final_emotions.dominant_emotion,
                    confidence=analysis_result.final_emotions.confidence
                )
            
            final_risk = None
            if analysis_result.final_risk:
                final_risk = RiskResponse(
                    level=analysis_result.final_risk.level,
                    score=analysis_result.final_risk.score,
                    factors=analysis_result.final_risk.factors,
                    recommendations=analysis_result.final_risk.recommendations,
                    confidence=analysis_result.final_risk.confidence
                )
            
            # Calculate overall confidence
            successful_predictions = [p for p in analysis_result.model_predictions if p.status == "success"]
            avg_confidence = (
                sum(p.confidence for p in successful_predictions) / len(successful_predictions)
                if successful_predictions else 0.0
            )
            
            history.append(AnalysisResponse(
                analysis_id=analysis_result.analysis_id,
                analysis_type=analysis_result.analysis_type,
                status=analysis_result.status,
                model_results=model_results,
                final_sentiment=final_sentiment,
                final_emotions=final_emotions,
                final_risk=final_risk,
                best_model=analysis_result.best_performing_model,
                successful_models=analysis_result.successful_models,
                failed_models=analysis_result.failed_models,
                processing_time=analysis_result.processing_time,
                created_at=analysis_result.created_at,
                confidence=avg_confidence
            ))
        
        return history
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting analysis history: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/models/status", response_model=ModelStatusResponse)
async def get_models_status():
    """
    Get status and information about available AI models
    
    Returns:
        ModelStatusResponse: Model status information
    """
    try:
        logger.info("🤖 API request to get models status")
        
        # Get model manager from app state
        from app.main import app
        if not hasattr(app.state, 'model_manager'):
            return ModelStatusResponse(
                available=False,
                models_loaded=[],
                total_models=4,
                error_message="AI models not initialized"
            )
        
        model_manager = app.state.model_manager
        
        # Get model information
        model_info = {
            "CNN": {
                "name": "Convolutional Neural Network",
                "accuracy": 86.0,
                "description": "Deep learning model for text classification",
                "status": "loaded" if model_manager.cnn_model else "failed"
            },
            "DNN": {
                "name": "Deep Neural Network",
                "accuracy": 90.0,
                "description": "Multi-layer neural network for sentiment analysis",
                "status": "loaded" if model_manager.dnn_model else "failed"
            },
            "CASTLE": {
                "name": "CASTLE Algorithm",
                "accuracy": 95.0,
                "description": "Advanced ensemble method for mental health detection",
                "status": "loaded" if model_manager.castle_model else "failed"
            },
            "MOON": {
                "name": "MOON Framework",
                "accuracy": 98.0,
                "description": "State-of-the-art model for psychological analysis",
                "status": "loaded" if model_manager.moon_model else "failed"
            }
        }
        
        loaded_models = [name for name, info in model_info.items() if info["status"] == "loaded"]
        
        return ModelStatusResponse(
            available=len(loaded_models) > 0,
            models_loaded=loaded_models,
            total_models=4,
            model_details=model_info,
            last_updated=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"❌ Error getting models status: {str(e)}")
        return ModelStatusResponse(
            available=False,
            models_loaded=[],
            total_models=4,
            error_message=f"Error checking model status: {str(e)}"
        )


@router.delete("/results/{analysis_id}")
async def delete_analysis_result(analysis_id: str):
    """
    Delete a specific analysis result
    
    Args:
        analysis_id: Analysis identifier
        
    Returns:
        dict: Success message
    """
    try:
        collection = await get_analysis_collection()
        result = await collection.delete_one({"analysis_id": analysis_id})
        
        if result.deleted_count == 0:
            raise HTTPException(
                status_code=404,
                detail=f"Analysis result with ID {analysis_id} not found"
            )
        
        logger.info(f"🗑️ Deleted analysis result: {analysis_id}")
        return {"message": f"Analysis result {analysis_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error deleting analysis result: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/statistics")
async def get_analysis_statistics():
    """
    Get analysis statistics and metrics
    
    Returns:
        dict: Statistics about analyses
    """
    try:
        collection = await get_analysis_collection()
        
        # Total analyses
        total_analyses = await collection.count_documents({})
        
        # Analyses by type
        twitter_count = await collection.count_documents({"analysis_type": "twitter"})
        csv_count = await collection.count_documents({"analysis_type": "csv"})
        
        # Recent analyses (last 7 days)
        recent_cutoff = datetime.utcnow() - timedelta(days=7)
        recent_count = await collection.count_documents({"created_at": {"$gte": recent_cutoff}})
        
        # Most used models
        pipeline = [
            {"$unwind": "$successful_models"},
            {"$group": {"_id": "$successful_models", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        
        model_usage = []
        async for doc in collection.aggregate(pipeline):
            model_usage.append({"model": doc["_id"], "usage_count": doc["count"]})
        
        # Average processing time
        avg_time_pipeline = [
            {"$group": {"_id": None, "avg_time": {"$avg": "$processing_time"}}}
        ]
        
        avg_processing_time = 0.0
        async for doc in collection.aggregate(avg_time_pipeline):
            avg_processing_time = doc.get("avg_time", 0.0)
        
        return {
            "total_analyses": total_analyses,
            "twitter_analyses": twitter_count,
            "csv_analyses": csv_count,
            "recent_analyses": recent_count,
            "most_used_models": model_usage,
            "average_processing_time": round(avg_processing_time, 2),
            "generated_at": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting analysis statistics: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
