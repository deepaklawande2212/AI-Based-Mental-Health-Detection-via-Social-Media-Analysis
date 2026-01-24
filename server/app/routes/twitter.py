"""
Twitter API Routes

This module provides REST API endpoints for Twitter data collection and analysis.

Endpoints:
- POST /collect/{username} - Collect Twitter data for a user
- GET /data/{username} - Get stored Twitter data
- POST /analyze/{username} - Analyze Twitter data
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Optional, List
from datetime import datetime
from loguru import logger

from app.models.request_models import TwitterAnalysisRequest, TwitterDataResponse, AnalysisResponse
from app.models.database_models import TwitterData, AnalysisResult
from app.services.twitter_api_service import twitter_service
from app.services.ai_models import ModelManager
from app.config.database import get_twitter_collection, get_analysis_collection
import uuid


router = APIRouter()


@router.post("/collect/{username}", response_model=TwitterDataResponse)
async def collect_twitter_data(
    username: str,
    max_tweets: Optional[int] = 100,
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Collect Twitter data for a specific user
    
    Args:
        username: Twitter username (without @)
        max_tweets: Maximum number of tweets to collect (default: 100)
        background_tasks: FastAPI background tasks
        
    Returns:
        TwitterDataResponse: Collection status and basic info
    """
    try:
        logger.info(f"🐦 API request to collect Twitter data for @{username}")
        
        # Validate username
        if not username or len(username) > 15:
            raise HTTPException(status_code=400, detail="Invalid username")
        
        # Clean username
        username = username.replace('@', '').strip().lower()
        
        # Check if data collection is already in progress or completed recently
        collection = await get_twitter_collection()
        existing = await collection.find_one({"username": username})
        
        if existing:
            twitter_data = TwitterData(**existing)
            
            # If data is fresh (less than 24 hours old), return existing data
            if (twitter_data.last_updated and 
                (datetime.utcnow() - twitter_data.last_updated).total_seconds() < 86400):
                
                return TwitterDataResponse(
                    username=twitter_data.username,
                    user_id=twitter_data.user_id,
                    display_name=twitter_data.display_name,
                    bio=twitter_data.bio,
                    followers_count=twitter_data.followers_count,
                    following_count=twitter_data.following_count,
                    tweet_count=twitter_data.tweet_count,
                    tweets_collected=twitter_data.total_tweets_collected,
                    collection_status=twitter_data.collection_status,
                    created_at=twitter_data.created_at
                )
        
        # Start data collection in background
        async def collect_data():
            try:
                await twitter_service.collect_user_data(username, max_tweets)
            except Exception as e:
                logger.error(f"❌ Background collection failed for @{username}: {str(e)}")
        
        background_tasks.add_task(collect_data)
        
        # Return immediate response
        return TwitterDataResponse(
            username=username,
            user_id=None,
            display_name=None,
            bio=None,
            followers_count=0,
            following_count=0,
            tweet_count=0,
            tweets_collected=0,
            collection_status="pending",
            created_at=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in collect_twitter_data: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/data/{username}", response_model=TwitterData)
async def get_twitter_data(username: str):
    """
    Get Twitter data for a specific username
    
    Args:
        username: Twitter username
        
    Returns:
        TwitterData: Twitter data including tweets and user info
    """
    try:
        logger.info(f"📊 API request to get Twitter data for @{username}")
        
        username = username.replace('@', '').strip().lower()
        
        try:
            # Get Twitter data from database
            collection = await get_twitter_collection()
            doc = await collection.find_one({"username": username})
            
            if doc:
                twitter_data = TwitterData(**doc)
                logger.info(f"✅ Retrieved Twitter data for @{username} from database")
                return twitter_data
            else:
                raise HTTPException(
                    status_code=404, 
                    detail=f"No Twitter data found for @{username}"
                )
                
        except Exception as e:
            logger.warning(f"⚠️ Could not access database: {str(e)}")
            logger.info("📝 Database not available, returning 404")
            raise HTTPException(
                status_code=404,
                detail=f"No Twitter data found for @{username} (database not available)"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting Twitter data: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/analyze/{username}", response_model=AnalysisResponse)
async def analyze_twitter_data(username: str):
    """
    Analyze Twitter data using AI models
    
    Args:
        username: Twitter username
        
    Returns:
        AnalysisResponse: Analysis results from all models
    """
    try:
        logger.info(f"🔮 API request to analyze Twitter data for @{username}")
        
        username = username.replace('@', '').strip().lower()
        
        # Get Twitter data
        collection = await get_twitter_collection()
        doc = await collection.find_one({"username": username})
        
        if not doc:
            raise HTTPException(
                status_code=404,
                detail=f"No Twitter data found for @{username}. Please collect data first."
            )
        
        twitter_data = TwitterData(**doc)
        
        if twitter_data.collection_status != "completed":
            raise HTTPException(
                status_code=400,
                detail="Twitter data collection is not completed yet"
            )
        
        if not twitter_data.tweets:
            raise HTTPException(
                status_code=400,
                detail="No tweets available for analysis"
            )
        
        # Combine all tweet texts for analysis
        tweet_texts = [tweet.text for tweet in twitter_data.tweets]
        combined_text = " ".join(tweet_texts)
        
        # Create content summary
        content_summary = f"Analyzed {len(tweet_texts)} tweets from @{username}"
        
        logger.info(f"✅ Collected {len(twitter_data.tweets)} real tweets for @{username}")
        
        # Limit text size for processing (max 100k characters)
        if len(combined_text) > 100000:
            combined_text = combined_text[:100000]
            logger.warning(f"⚠️ Text truncated for analysis (original: {len(' '.join(tweet_texts))} chars)")
        
        # Get model manager from app state
        from app.main import app
        if not hasattr(app.state, 'model_manager'):
            raise HTTPException(
                status_code=503,
                detail="AI models not available"
            )
        
        model_manager = app.state.model_manager
        
        # Run AI analysis
        model_predictions = await model_manager.predict_all_models(combined_text)
        
        if not model_predictions:
            raise HTTPException(
                status_code=500,
                detail="No model predictions were successful"
            )
        
        # Calculate aggregated results (use best performing model)
        best_model = max(model_predictions, key=lambda x: x.accuracy if x.status == "success" else 0)
        
        # Create analysis result
        analysis_id = str(uuid.uuid4())
        
        # Convert model predictions to response format
        from app.models.request_models import ModelResultResponse, SentimentResponse, EmotionResponse, RiskResponse
        
        model_results = []
        for pred in model_predictions:
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
        if best_model.sentiment:
            final_sentiment = SentimentResponse(
                positive=best_model.sentiment.positive,
                negative=best_model.sentiment.negative,
                neutral=best_model.sentiment.neutral,
                overall=best_model.sentiment.overall,
                confidence=best_model.sentiment.confidence
            )
        
        final_emotions = None
        if best_model.emotions:
            final_emotions = EmotionResponse(
                joy=best_model.emotions.joy,
                sadness=best_model.emotions.sadness,
                anger=best_model.emotions.anger,
                fear=best_model.emotions.fear,
                surprise=best_model.emotions.surprise,
                disgust=best_model.emotions.disgust,
                dominant_emotion=best_model.emotions.dominant_emotion,
                confidence=best_model.emotions.confidence
            )
        
        final_risk = None
        if best_model.risk:
            final_risk = RiskResponse(
                level=best_model.risk.level,
                score=best_model.risk.score,
                factors=best_model.risk.factors,
                recommendations=best_model.risk.recommendations,
                confidence=best_model.risk.confidence
            )
        
        # Calculate realistic confidence
        from app.routes.csv import _calculate_realistic_confidence
        
        return AnalysisResponse(
            analysis_id=analysis_id,
            analysis_type="twitter",
            analyzed_content=tweet_texts,  # Include the actual tweets
            content_summary=content_summary,  # Include summary
            model_results=model_results,
            final_sentiment=final_sentiment,
            final_emotions=final_emotions,
            final_risk=final_risk,
            best_model=best_model.model_name,
            successful_models=[p.model_name for p in model_predictions if p.status == "success"],
            failed_models=[p.model_name for p in model_predictions if p.status == "failed"],
            processing_time=sum(p.prediction_time for p in model_predictions),
            status="completed",
            created_at=datetime.utcnow(),
            confidence=_calculate_realistic_confidence(model_results)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error analyzing Twitter data: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/users", response_model=List[TwitterDataResponse])
async def list_twitter_users(limit: int = 50, skip: int = 0):
    """
    List all stored Twitter users
    
    Args:
        limit: Maximum number of users to return
        skip: Number of users to skip
        
    Returns:
        List[TwitterDataResponse]: List of Twitter users
    """
    try:
        collection = await get_twitter_collection()
        
        cursor = collection.find({}).sort("created_at", -1).skip(skip).limit(limit)
        docs = await cursor.to_list(length=limit)
        
        users = []
        for doc in docs:
            twitter_data = TwitterData(**doc)
            users.append(TwitterDataResponse(
                username=twitter_data.username,
                user_id=twitter_data.user_id,
                display_name=twitter_data.display_name,
                bio=twitter_data.bio,
                followers_count=twitter_data.followers_count,
                following_count=twitter_data.following_count,
                tweet_count=twitter_data.tweet_count,
                tweets_collected=twitter_data.total_tweets_collected,
                collection_status=twitter_data.collection_status,
                created_at=twitter_data.created_at
            ))
        
        return users
        
    except Exception as e:
        logger.error(f"❌ Error listing Twitter users: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/data/{username}")
async def delete_twitter_data(username: str):
    """
    Delete stored Twitter data for a user
    
    Args:
        username: Twitter username
        
    Returns:
        dict: Success message
    """
    try:
        username = username.replace('@', '').strip().lower()
        
        collection = await get_twitter_collection()
        result = await collection.delete_one({"username": username})
        
        if result.deleted_count == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No Twitter data found for @{username}"
            )
        
        logger.info(f"🗑️ Deleted Twitter data for @{username}")
        return {"message": f"Twitter data for @{username} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error deleting Twitter data: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_twitter_direct(request: TwitterAnalysisRequest):
    """
    Analyze Twitter data directly using a request body
    
    Args:
        request: TwitterAnalysisRequest containing username and options
        
    Returns:
        AnalysisResponse: Analysis results from all models
    """
    try:
        logger.info(f"🔮 Direct Twitter analysis request for @{request.username}")
        
        username = request.username.replace('@', '').strip().lower()
        max_tweets = request.max_tweets or 15
        include_retweets = request.include_retweets if request.include_retweets is not None else True
        
        # Collect real Twitter data using the service
        try:
            logger.info(f"📥 Collecting real Twitter data for @{username}")
            twitter_data = await twitter_service.collect_user_data(
                username=username,
                max_tweets=max_tweets
            )
            
            if not twitter_data.tweets or len(twitter_data.tweets) == 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"No tweets found for @{username}. The account might be private or have no tweets."
                )
            
            # Combine all tweet texts for analysis
            tweet_texts = [tweet.text for tweet in twitter_data.tweets]
            combined_text = " ".join(tweet_texts)
            
            logger.info(f"✅ Collected {len(twitter_data.tweets)} real tweets for @{username}")
            
        except Exception as e:
            logger.error(f"❌ Failed to collect real Twitter data for @{username}: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to collect Twitter data: {str(e)}"
            )
        
        # Limit text size for processing (max 100k characters)
        if len(combined_text) > 100000:
            combined_text = combined_text[:100000]
            logger.warning(f"⚠️ Text truncated for analysis (original: {len(' '.join(tweet_texts))} chars)")
        
        # Get model manager from app state
        from app.main import app
        if not hasattr(app.state, 'model_manager'):
            raise HTTPException(
                status_code=503,
                detail="AI models not available"
            )
        
        model_manager = app.state.model_manager
        
        # Run AI analysis
        model_predictions = await model_manager.predict_all_models(combined_text)
        
        if not model_predictions:
            raise HTTPException(
                status_code=500,
                detail="No model predictions were successful"
            )
        
        # Calculate aggregated results (use best performing model)
        best_model = max(model_predictions, key=lambda x: x.accuracy if x.status == "success" else 0)
        
        # Create analysis result
        analysis_id = str(uuid.uuid4())
        
        # Convert model predictions to response format
        from app.models.request_models import ModelResultResponse, SentimentResponse, EmotionResponse, RiskResponse
        
        model_results = []
        for pred in model_predictions:
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
        if best_model.sentiment:
            final_sentiment = SentimentResponse(
                positive=best_model.sentiment.positive,
                negative=best_model.sentiment.negative,
                neutral=best_model.sentiment.neutral,
                overall=best_model.sentiment.overall,
                confidence=best_model.sentiment.confidence
            )
        
        final_emotions = None
        if best_model.emotions:
            final_emotions = EmotionResponse(
                joy=best_model.emotions.joy,
                sadness=best_model.emotions.sadness,
                anger=best_model.emotions.anger,
                fear=best_model.emotions.fear,
                surprise=best_model.emotions.surprise,
                disgust=best_model.emotions.disgust,
                dominant_emotion=best_model.emotions.dominant_emotion,
                confidence=best_model.emotions.confidence
            )
        
        final_risk = None
        if best_model.risk:
            final_risk = RiskResponse(
                level=best_model.risk.level,
                score=best_model.risk.score,
                factors=best_model.risk.factors,
                recommendations=best_model.risk.recommendations,
                confidence=best_model.risk.confidence
            )
        
        # Calculate realistic confidence
        from app.routes.csv import _calculate_realistic_confidence
        
        return AnalysisResponse(
            analysis_id=analysis_id,
            analysis_type="twitter",
            status="completed",
            model_results=model_results,
            final_sentiment=final_sentiment,
            final_emotions=final_emotions,
            final_risk=final_risk,
            best_model=best_model.model_name,
            successful_models=[p.model_name for p in model_predictions if p.status == "success"],
            failed_models=[p.model_name for p in model_predictions if p.status == "failed"],
            processing_time=sum(p.prediction_time for p in model_predictions),
            created_at=datetime.utcnow(),
            confidence=best_model.confidence if best_model else 0.0
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in direct Twitter analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/analyze-direct", response_model=AnalysisResponse)
async def analyze_twitter_direct(request: TwitterAnalysisRequest):
    """
    Analyze Twitter data directly without database storage
    
    Args:
        request: TwitterAnalysisRequest containing username and options
        
    Returns:
        AnalysisResponse: Analysis results from all models
    """
    try:
        logger.info(f"🐦 Direct Twitter analysis request for @{request.username}")
        
        if not request.username or len(request.username) > 15:
            raise HTTPException(
                status_code=400,
                detail="Invalid username"
            )
        
        # Clean username
        username = request.username.replace('@', '').strip().lower()
        
        # Get model manager from app state
        from app.main import app
        if not hasattr(app.state, 'model_manager'):
            raise HTTPException(
                status_code=503,
                detail="AI models not available"
            )
        
        model_manager = app.state.model_manager
        
        # Initialize variables
        tweet_texts = []
        content_summary = ""
        combined_text = ""
        
        # Collect real Twitter data using the service
        try:
            logger.info(f"📥 Collecting real Twitter data for @{username}")
            twitter_data = await twitter_service.collect_user_data(
                username=username,
                max_tweets=request.max_tweets or 15
            )
            
            if not twitter_data.tweets or len(twitter_data.tweets) == 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"No tweets found for @{username}. The account might be private or have no tweets."
                )
            
            # Combine all tweet texts for analysis
            tweet_texts = [tweet.text for tweet in twitter_data.tweets]
            combined_text = " ".join(tweet_texts)
            
            # Create content summary
            content_summary = f"Analyzed {len(tweet_texts)} tweets from @{username}"
            
            logger.info(f"✅ Collected {len(twitter_data.tweets)} real tweets for @{username}")
            
        except Exception as e:
            logger.error(f"❌ Failed to collect real Twitter data for @{username}: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to collect Twitter data: {str(e)}"
            )
        
        # Debug logging to see what we're analyzing
        logger.info(f"🔍 Debug: Analyzing text: {combined_text[:200]}...")
        logger.info(f"🔍 Debug: Tweet texts: {tweet_texts}")
        logger.info(f"🔍 Debug: Content summary: {content_summary}")
        
        # Run AI analysis
        model_predictions = await model_manager.predict_all_models(combined_text)
        
        if not model_predictions:
            raise HTTPException(
                status_code=500,
                detail="No model predictions were successful"
            )
        
        # Calculate aggregated results (use best performing model)
        best_model = max(model_predictions, key=lambda x: x.accuracy if x.status == "success" else 0)
        
        # Create analysis result
        analysis_id = str(uuid.uuid4())
        
        # Convert model predictions to response format
        from app.models.request_models import ModelResultResponse, SentimentResponse, EmotionResponse, RiskResponse
        
        model_results = []
        for pred in model_predictions:
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
        if best_model.sentiment:
            final_sentiment = SentimentResponse(
                positive=best_model.sentiment.positive,
                negative=best_model.sentiment.negative,
                neutral=best_model.sentiment.neutral,
                overall=best_model.sentiment.overall,
                confidence=best_model.sentiment.confidence
            )
        
        final_emotions = None
        if best_model.emotions:
            final_emotions = EmotionResponse(
                joy=best_model.emotions.joy,
                sadness=best_model.emotions.sadness,
                anger=best_model.emotions.anger,
                fear=best_model.emotions.fear,
                surprise=best_model.emotions.surprise,
                disgust=best_model.emotions.disgust,
                dominant_emotion=best_model.emotions.dominant_emotion,
                confidence=best_model.emotions.confidence
            )
        
        final_risk = None
        if best_model.risk:
            final_risk = RiskResponse(
                level=best_model.risk.level,
                score=best_model.risk.score,
                factors=best_model.risk.factors,
                recommendations=best_model.risk.recommendations,
                confidence=best_model.risk.confidence
            )
        
        # Calculate realistic confidence
        from app.routes.csv import _calculate_realistic_confidence
        
        return AnalysisResponse(
            analysis_id=analysis_id,
            analysis_type="twitter",
            analyzed_content=tweet_texts,  # Include the actual tweets
            content_summary=content_summary,  # Include summary
            model_results=model_results,
            final_sentiment=final_sentiment,
            final_emotions=final_emotions,
            final_risk=final_risk,
            best_model=best_model.model_name,
            successful_models=[p.model_name for p in model_predictions if p.status == "success"],
            failed_models=[p.model_name for p in model_predictions if p.status == "failed"],
            processing_time=sum(p.prediction_time for p in model_predictions),
            status="completed",
            created_at=datetime.utcnow(),
            confidence=_calculate_realistic_confidence(model_results)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in direct Twitter analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/analyze-csv", response_model=AnalysisResponse)
async def analyze_twitter_as_csv(request: TwitterAnalysisRequest):
    """
    Analyze Twitter data by converting tweets to CSV and using existing CSV analysis
    """
    try:
        logger.info(f"📊 Twitter CSV analysis request for @{request.username}")
        
        username = request.username.replace('@', '').strip().lower()
        max_tweets = request.max_tweets or 15
        
        # Get model manager from app state
        from app.main import app
        if not hasattr(app.state, 'model_manager'):
            raise HTTPException(
                status_code=503,
                detail="AI models not available"
            )
        
        model_manager = app.state.model_manager
        
        # Fetch real tweets using the service
        try:
            logger.info(f"📥 Fetching real tweets for @{username}")
            twitter_data = await twitter_service.collect_user_data(
                username=username,
                max_tweets=max_tweets
            )
            
            if not twitter_data.tweets or len(twitter_data.tweets) == 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"No tweets found for @{username}. The account might be private or have no tweets."
                )
            
            # Convert tweets to CSV format (for conceptual understanding, actual analysis uses combined text)
            import csv
            import io
            import tempfile
            import os
            
            def detect_tweet_status(text: str) -> str:
                """Detect status based on tweet content"""
                text_lower = text.lower()
                
                # Crisis indicators
                crisis_keywords = [
                    'suicide', 'kill myself', 'end it all', 'want to die', 'better off dead',
                    'hurt myself', 'self harm', 'cut myself', 'overdose', 'overdosing',
                    'no point living', 'life is worthless', 'worthless', 'hopeless',
                    'cannot take this', 'cannot bear', 'too much pain', 'end my life',
                    'thinking of ending', 'thoughts of suicide', 'suicidal thoughts'
                ]
                
                # Negative indicators
                negative_keywords = [
                    'depressed', 'depression', 'sad', 'sadness', 'hopeless', 'hopelessness',
                    'anxiety', 'anxious', 'stress', 'stressed', 'overwhelmed', 'overwhelming',
                    'lonely', 'loneliness', 'isolated', 'isolation', 'worthless', 'useless',
                    'failure', 'failed', 'disappointed', 'disappointment', 'frustrated',
                    'angry', 'anger', 'hate', 'hatred', 'miserable', 'misery',
                    'tired', 'exhausted', 'burned out', 'burnout', 'give up', 'giving up'
                ]
                
                # Positive indicators
                positive_keywords = [
                    'happy', 'happiness', 'joy', 'excited', 'excitement', 'thrilled',
                    'grateful', 'gratitude', 'blessed', 'blessing', 'amazing', 'wonderful',
                    'fantastic', 'great', 'good', 'excellent', 'perfect', 'love', 'loving',
                    'proud', 'pride', 'accomplished', 'success', 'successful', 'achievement',
                    'motivated', 'motivation', 'inspired', 'inspiration', 'hope', 'hopeful',
                    'optimistic', 'optimism', 'positive', 'positivity', 'smile', 'smiling'
                ]
                
                # Check for crisis first (highest priority)
                for keyword in crisis_keywords:
                    if keyword in text_lower:
                        return "Crisis"
                
                # Count positive and negative keywords
                positive_count = sum(1 for keyword in positive_keywords if keyword in text_lower)
                negative_count = sum(1 for keyword in negative_keywords if keyword in text_lower)
                
                # Determine status based on keyword balance
                if positive_count > negative_count:
                    return "Positive"
                elif negative_count > positive_count:
                    return "Negative"
                else:
                    return "Neutral"
            
            # Create CSV content in the correct format (text,status)
            csv_content = io.StringIO()
            csv_writer = csv.writer(csv_content)
            
            # Write header - match the expected format
            csv_writer.writerow(['text', 'status'])
            
            # Write tweet data - only text column, status is optional
            tweet_texts = []
            for tweet in twitter_data.tweets:
                status = detect_tweet_status(tweet.text)
                csv_writer.writerow([
                    tweet.text,
                    status  # Use detected status instead of 'Unknown'
                ])
                tweet_texts.append(tweet.text)
            
            csv_content.seek(0)
            csv_data = csv_content.getvalue()
            
            logger.info(f"✅ Converted {len(twitter_data.tweets)} tweets to CSV format (text,status) with intelligent status detection")
            
            # Create a temporary CSV file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
                temp_file.write(csv_data)
                temp_file_path = temp_file.name
            
            try:
                # Use existing CSV analysis logic directly
                from app.routes.csv import _calculate_realistic_confidence
                from app.models.request_models import ModelResultResponse, SentimentResponse, EmotionResponse, RiskResponse
                
                # Get model manager from app state
                model_manager = app.state.model_manager
                
                # Combine all tweet texts for analysis
                combined_text = " ".join(tweet_texts)
                
                # Run AI analysis
                model_predictions = await model_manager.predict_all_models(combined_text)
                
                if not model_predictions:
                    raise HTTPException(
                        status_code=500,
                        detail="No model predictions were successful"
                    )
                
                # Calculate aggregated results (use best performing model)
                best_model = max(model_predictions, key=lambda x: x.accuracy if x.status == "success" else 0)
                
                # Create analysis result
                analysis_id = str(uuid.uuid4())
                
                # Convert model predictions to response format
                model_results = []
                for pred in model_predictions:
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
                if best_model.sentiment:
                    final_sentiment = SentimentResponse(
                        positive=best_model.sentiment.positive,
                        negative=best_model.sentiment.negative,
                        neutral=best_model.sentiment.neutral,
                        overall=best_model.sentiment.overall,
                        confidence=best_model.sentiment.confidence
                    )
                
                final_emotions = None
                if best_model.emotions:
                    final_emotions = EmotionResponse(
                        joy=best_model.emotions.joy,
                        sadness=best_model.emotions.sadness,
                        anger=best_model.emotions.anger,
                        fear=best_model.emotions.fear,
                        surprise=best_model.emotions.surprise,
                        disgust=best_model.emotions.disgust,
                        dominant_emotion=best_model.emotions.dominant_emotion,
                        confidence=best_model.emotions.confidence
                    )
                
                final_risk = None
                if best_model.risk:
                    final_risk = RiskResponse(
                        level=best_model.risk.level,
                        score=best_model.risk.score,
                        factors=best_model.risk.factors,
                        recommendations=best_model.risk.recommendations,
                        confidence=best_model.risk.confidence
                    )
                
                # Calculate realistic confidence
                confidence = _calculate_realistic_confidence(model_results)
                
                analysis_result = AnalysisResponse(
                    analysis_id=analysis_id,
                    analysis_type="twitter",
                    analyzed_content=tweet_texts,
                    content_summary=f"Analyzed {len(tweet_texts)} real tweets from @{username}",
                    model_results=model_results,
                    final_sentiment=final_sentiment,
                    final_emotions=final_emotions,
                    final_risk=final_risk,
                    best_model=best_model.model_name,
                    successful_models=[p.model_name for p in model_predictions if p.status == "success"],
                    failed_models=[p.model_name for p in model_predictions if p.status == "failed"],
                    processing_time=sum(p.prediction_time for p in model_predictions),
                    status="completed",
                    created_at=datetime.utcnow(),
                    confidence=confidence
                )
                
                logger.success(f"✅ Successfully analyzed {len(tweet_texts)} real tweets from @{username}")
                return analysis_result
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"❌ Error in Twitter CSV analysis: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error analyzing Twitter data: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in Twitter CSV analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
