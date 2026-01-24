"""
Request and Response Models for API Endpoints

This module defines Pydantic models for API request/response validation.
These models ensure proper data validation and documentation.

Phase 1: API Interface Models
- Request models for incoming data
- Response models for API responses
- Validation schemas
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict
from enum import Enum


class AnalysisType(str, Enum):
    """Analysis type enumeration"""
    TWITTER = "twitter"
    CSV = "csv"
    TEXT = "text"


class RiskLevel(str, Enum):
    """Risk level enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ModelName(str, Enum):
    """AI model name enumeration"""
    CNN = "cnn"
    DNN = "dnn"
    CASTLE = "castle"
    MOON = "moon"
    DECISION_TREE = "decision_tree"
    LSTM = "lstm"
    RNN = "rnn"
    BERT = "bert"


# Request Models
class TwitterAnalysisRequest(BaseModel):
    """Request model for Twitter data analysis"""
    
    username: str = Field(..., min_length=1, max_length=15, description="Twitter username")
    max_tweets: Optional[int] = Field(default=100, ge=1, le=200, description="Maximum tweets to collect")
    include_retweets: bool = Field(default=False, description="Include retweets in analysis")
    
    @field_validator('username')
    @classmethod
    def validate_username(cls, v):
        # Remove @ if present
        username = v.replace('@', '').strip()
        if not username.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Invalid Twitter username format')
        return username


class TextAnalysisRequest(BaseModel):
    """Request model for direct text analysis"""
    
    text: str = Field(..., min_length=10, max_length=10000, description="Text to analyze")
    language: Optional[str] = Field(default="en", description="Text language")
    
    @field_validator('text')
    @classmethod
    def validate_text(cls, v):
        # Basic text validation
        if not v.strip():
            raise ValueError('Text cannot be empty')
        return v.strip()


class CSVAnalysisRequest(BaseModel):
    """Request model for CSV file analysis"""
    
    file_id: str = Field(..., description="Uploaded CSV file ID")
    text_column: Optional[str] = Field(default=None, description="Column name containing text data")
    max_rows: Optional[int] = Field(default=1000, ge=1, le=5000, description="Maximum rows to analyze")


# Response Models
class SentimentResponse(BaseModel):
    """Sentiment analysis response"""
    
    positive: float = Field(..., description="Positive sentiment percentage")
    negative: float = Field(..., description="Negative sentiment percentage") 
    neutral: float = Field(..., description="Neutral sentiment percentage")
    overall: str = Field(..., description="Overall sentiment")
    confidence: float = Field(..., description="Confidence score")


class EmotionResponse(BaseModel):
    """Emotion detection response"""
    
    joy: float = Field(..., description="Joy emotion score")
    sadness: float = Field(..., description="Sadness emotion score")
    anger: float = Field(..., description="Anger emotion score")
    fear: float = Field(..., description="Fear emotion score")
    surprise: float = Field(..., description="Surprise emotion score")
    disgust: float = Field(..., description="Disgust emotion score")
    dominant_emotion: str = Field(..., description="Dominant emotion")
    confidence: float = Field(..., description="Confidence score")


class RiskResponse(BaseModel):
    """Risk assessment response"""
    
    level: RiskLevel = Field(..., description="Risk level")
    score: float = Field(..., description="Risk score")
    factors: List[str] = Field(..., description="Risk factors")
    recommendations: List[str] = Field(..., description="Recommendations")
    confidence: float = Field(..., description="Confidence score")


class ModelResultResponse(BaseModel):
    """Individual model result response"""
    
    model_name: ModelName = Field(..., description="Model name")
    accuracy: float = Field(..., description="Model accuracy")
    processing_time: float = Field(..., description="Processing time in seconds")
    status: str = Field(..., description="Processing status")
    confidence: float = Field(..., description="Overall confidence")
    sentiment: Optional[SentimentResponse] = Field(default=None, description="Sentiment results")
    emotions: Optional[EmotionResponse] = Field(default=None, description="Emotion results")
    risk: Optional[RiskResponse] = Field(default=None, description="Risk results")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")


class AnalysisResponse(BaseModel):
    """Complete analysis response"""
    
    analysis_id: str = Field(..., description="Analysis identifier")
    analysis_type: AnalysisType = Field(..., description="Analysis type")
    status: str = Field(..., description="Analysis status")
    
    # Analyzed content (for transparency)
    analyzed_content: Optional[List[str]] = Field(default=None, description="List of analyzed texts/tweets")
    content_summary: Optional[str] = Field(default=None, description="Summary of analyzed content")
    
    # Model results
    model_results: List[ModelResultResponse] = Field(..., description="Results from all models")
    
    # Final aggregated results
    final_sentiment: Optional[SentimentResponse] = Field(default=None, description="Final sentiment")
    final_emotions: Optional[EmotionResponse] = Field(default=None, description="Final emotions")
    final_risk: Optional[RiskResponse] = Field(default=None, description="Final risk assessment")
    
    # Performance summary
    best_model: Optional[str] = Field(default=None, description="Best performing model")
    successful_models: List[str] = Field(..., description="Successful models")
    failed_models: List[str] = Field(..., description="Failed models")
    
    # Metadata
    processing_time: float = Field(..., description="Total processing time")
    created_at: datetime = Field(..., description="Analysis timestamp")
    confidence: float = Field(..., description="Overall confidence")


class TwitterDataResponse(BaseModel):
    """Twitter data collection response"""
    
    username: str = Field(..., description="Twitter username")
    user_id: Optional[str] = Field(default=None, description="Twitter user ID")
    display_name: Optional[str] = Field(default=None, description="Display name")
    bio: Optional[str] = Field(default=None, description="User bio")
    followers_count: int = Field(..., description="Followers count")
    following_count: int = Field(..., description="Following count")
    tweet_count: int = Field(..., description="Total tweets")
    tweets_collected: int = Field(..., description="Tweets collected")
    collection_status: str = Field(..., description="Collection status")
    created_at: datetime = Field(..., description="Collection timestamp")


class CSVUploadResponse(BaseModel):
    """CSV file upload response"""
    
    file_id: str = Field(..., description="File identifier")
    filename: str = Field(..., description="Original filename")
    file_size: Optional[int] = Field(default=None, description="File size in bytes")
    total_rows: Optional[int] = Field(default=None, description="Total number of rows")
    valid_rows: Optional[int] = Field(default=None, description="Number of valid text rows")
    row_count: Optional[int] = Field(default=None, description="Number of rows (alias for total_rows)")
    columns: Optional[List[str]] = Field(default=None, description="Column names") 
    text_column: Optional[str] = Field(default=None, description="Main text column")
    upload_date: Optional[datetime] = Field(default=None, description="Upload timestamp")
    created_at: Optional[datetime] = Field(default=None, description="Creation timestamp")
    processing_status: Optional[str] = Field(default=None, description="Processing status")
    status: Optional[str] = Field(default=None, description="Processing status (alias)")
    description: Optional[str] = Field(default=None, description="File description")


class ModelStatusResponse(BaseModel):
    """Model status response"""
    
    model_name: str = Field(..., description="Model name")
    loaded: bool = Field(..., description="Is model loaded")
    accuracy: float = Field(..., description="Model accuracy")
    description: str = Field(..., description="Model description")
    last_updated: Optional[datetime] = Field(default=None, description="Last update")


class HealthCheckResponse(BaseModel):
    """Health check response"""
    
    status: str = Field(..., description="Service status")
    database: str = Field(..., description="Database status")
    models: Dict[str, bool] = Field(..., description="Model status")
    timestamp: datetime = Field(..., description="Check timestamp")


class ErrorResponse(BaseModel):
    """Error response model"""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")


class SuccessResponse(BaseModel):
    """Generic success response"""
    
    message: str = Field(..., description="Success message")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Response data")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")


# File upload models
class FileUploadResponse(BaseModel):
    """File upload response"""
    
    file_id: str = Field(..., description="Uploaded file ID")
    filename: str = Field(..., description="Original filename")
    size: int = Field(..., description="File size in bytes")
    content_type: str = Field(..., description="File content type")
    upload_date: datetime = Field(..., description="Upload timestamp")


# Pagination models
class PaginationParams(BaseModel):
    """Pagination parameters"""
    
    page: int = Field(default=1, ge=1, description="Page number")
    limit: int = Field(default=10, ge=1, le=100, description="Items per page")
    sort_by: Optional[str] = Field(default="created_at", description="Sort field")
    sort_order: Optional[str] = Field(default="desc", description="Sort order")
    
    @field_validator('sort_order')
    @classmethod
    def validate_sort_order(cls, v):
        if v not in ["asc", "desc"]:
            raise ValueError('Sort order must be either "asc" or "desc"')
        return v


class PaginatedResponse(BaseModel):
    """Paginated response wrapper"""
    
    data: List[Any] = Field(..., description="Response data")
    total: int = Field(..., description="Total items")
    page: int = Field(..., description="Current page")
    limit: int = Field(..., description="Items per page")
    total_pages: int = Field(..., description="Total pages")
    has_next: bool = Field(..., description="Has next page")
    has_prev: bool = Field(..., description="Has previous page")
