"""
Database Models for MongoDB Collections

This module defines the data structure for MongoDB documents.
These models represent the schema for our collections.

Phase 1: Data Storage Models
- TwitterData: Stores Twitter user data and tweets
- CSVData: Stores uploaded CSV files and metadata
- AnalysisResult: Stores AI analysis results
"""

from datetime import datetime
from typing import List, Optional, Dict, Any, Annotated
from pydantic import BaseModel, Field, ConfigDict, BeforeValidator
from bson import ObjectId


# PyObjectId for Pydantic v2
PyObjectId = Annotated[str, BeforeValidator(str)]


class Tweet(BaseModel):
    """Individual tweet data structure"""
    
    id: str = Field(..., description="Tweet ID")
    text: str = Field(..., description="Tweet content")
    created_at: datetime = Field(..., description="Tweet creation date")
    retweet_count: int = Field(default=0, description="Number of retweets")
    like_count: int = Field(default=0, description="Number of likes")
    reply_count: int = Field(default=0, description="Number of replies")
    language: Optional[str] = Field(default="en", description="Tweet language")
    is_retweet: bool = Field(default=False, description="Is this a retweet")
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_encoders={ObjectId: str}
    )


class TwitterData(BaseModel):
    """Twitter user data and tweets collection model"""
    
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    username: str = Field(..., description="Twitter username")
    user_id: Optional[str] = Field(default=None, description="Twitter user ID")
    display_name: Optional[str] = Field(default=None, description="User display name")
    bio: Optional[str] = Field(default=None, description="User bio/description")
    followers_count: int = Field(default=0, description="Number of followers")
    following_count: int = Field(default=0, description="Number of following")
    tweet_count: int = Field(default=0, description="Total tweets by user")
    account_created: Optional[datetime] = Field(default=None, description="Account creation date")
    
    # Tweet data
    tweets: List[Tweet] = Field(default=[], description="Collected tweets")
    total_tweets_collected: int = Field(default=0, description="Number of tweets collected")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Data collection date")
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Last update date")
    collection_status: str = Field(default="pending", description="Collection status: pending/completed/failed")
    error_message: Optional[str] = Field(default=None, description="Error message if collection failed")
    
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_encoders={ObjectId: str}
    )


class CSVData(BaseModel):
    """CSV file data collection model"""
    
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    file_id: str = Field(..., description="Unique file identifier")
    filename: str = Field(..., description="Original filename")
    file_size: int = Field(..., description="File size in bytes")
    content_type: str = Field(..., description="File content type")
    
    # File content
    headers: List[str] = Field(default=[], description="CSV column headers")
    data: List[Dict[str, Any]] = Field(default=[], description="CSV data rows")
    row_count: int = Field(default=0, description="Number of data rows")
    total_rows: Optional[int] = Field(default=None, description="Total number of rows (alias for row_count)")
    valid_text_rows: Optional[int] = Field(default=None, description="Number of valid text rows")
    
    # Processing info
    text_column: Optional[str] = Field(default=None, description="Main text column for analysis")
    processed_texts: List[str] = Field(default=[], description="Extracted texts for analysis")
    description: Optional[str] = Field(default=None, description="File description")
    
    # Metadata
    upload_date: datetime = Field(default_factory=datetime.utcnow, description="File upload date")
    created_at: Optional[datetime] = Field(default=None, description="Creation date (alias for upload_date)")
    processed_at: Optional[datetime] = Field(default=None, description="Processing completion date")
    processing_status: str = Field(default="pending", description="Processing status: pending/completed/failed")
    error_message: Optional[str] = Field(default=None, description="Error message if processing failed")
    
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_encoders={ObjectId: str}
    )


class SentimentScore(BaseModel):
    """Sentiment analysis results"""
    
    positive: float = Field(..., description="Positive sentiment score (0-100)")
    negative: float = Field(..., description="Negative sentiment score (0-100)")
    neutral: float = Field(..., description="Neutral sentiment score (0-100)")
    overall: str = Field(..., description="Overall sentiment: positive/negative/neutral")
    confidence: float = Field(..., description="Confidence score (0-100)")


class EmotionScore(BaseModel):
    """Emotion detection results"""
    
    joy: float = Field(default=0.0, description="Joy emotion score (0-100)")
    sadness: float = Field(default=0.0, description="Sadness emotion score (0-100)")
    anger: float = Field(default=0.0, description="Anger emotion score (0-100)")
    fear: float = Field(default=0.0, description="Fear emotion score (0-100)")
    surprise: float = Field(default=0.0, description="Surprise emotion score (0-100)")
    disgust: float = Field(default=0.0, description="Disgust emotion score (0-100)")
    dominant_emotion: str = Field(..., description="Dominant emotion")
    confidence: float = Field(..., description="Confidence score (0-100)")


class RiskAssessment(BaseModel):
    """Risk assessment results"""
    
    level: str = Field(..., description="Risk level: low/medium/high")
    score: float = Field(..., description="Risk score (0-100)")
    factors: List[str] = Field(default=[], description="Contributing risk factors")
    recommendations: List[str] = Field(default=[], description="Recommended actions")
    confidence: float = Field(..., description="Confidence score (0-100)")


class ModelPrediction(BaseModel):
    """Individual AI model prediction"""
    
    model_name: str = Field(..., description="Model name: CNN/DNN/CASTLE/MOON")
    accuracy: float = Field(..., description="Model accuracy percentage")
    prediction_time: float = Field(..., description="Prediction time in seconds")
    confidence: float = Field(..., description="Prediction confidence (0-100)")
    status: str = Field(..., description="Prediction status: success/failed")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    
    # Model-specific results
    sentiment: Optional[SentimentScore] = Field(default=None, description="Sentiment analysis results")
    emotions: Optional[EmotionScore] = Field(default=None, description="Emotion detection results")
    risk: Optional[RiskAssessment] = Field(default=None, description="Risk assessment results")


class AnalysisResult(BaseModel):
    """Complete analysis results collection model"""
    
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    analysis_id: str = Field(..., description="Unique analysis identifier")
    analysis_type: str = Field(..., description="Analysis type: twitter/csv/text")
    
    # Source data references
    username: Optional[str] = Field(default=None, description="Twitter username (for Twitter analysis)")
    file_id: Optional[str] = Field(default=None, description="File ID (for CSV analysis)")
    input_text: Optional[str] = Field(default=None, description="Direct text input")
    
    # Analysis results from all models
    model_predictions: List[ModelPrediction] = Field(default=[], description="Predictions from all models")
    
    # Aggregated results (best performing model or ensemble)
    final_sentiment: Optional[SentimentScore] = Field(default=None, description="Final sentiment analysis")
    final_emotions: Optional[EmotionScore] = Field(default=None, description="Final emotion detection")
    final_risk: Optional[RiskAssessment] = Field(default=None, description="Final risk assessment")
    
    # Model performance summary
    best_performing_model: Optional[str] = Field(default=None, description="Best performing model")
    successful_models: List[str] = Field(default=[], description="Successfully executed models")
    failed_models: List[str] = Field(default=[], description="Failed models")
    
    # Processing metadata
    processing_time: float = Field(default=0.0, description="Total processing time in seconds")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Analysis creation date")
    status: str = Field(default="pending", description="Analysis status: pending/completed/failed")
    error_message: Optional[str] = Field(default=None, description="Error message if analysis failed")
    
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_encoders={ObjectId: str, datetime: lambda v: v.isoformat()}
    )
