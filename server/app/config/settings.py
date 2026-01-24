"""
Settings Configuration

This module handles all environment variables and application settings.
It uses Pydantic Settings for type validation and environment variable loading.

Phase 1: Basic Configuration
- Database settings
- API keys
- Server configuration
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import List
import os
from pathlib import Path


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables
    """
    
    # Environment
    ENVIRONMENT: str = Field(default="development", description="Application environment")
    DEBUG: bool = Field(default=True, description="Debug mode")
    
    # Server Configuration
    HOST: str = Field(default="0.0.0.0", description="Server host")
    PORT: int = Field(default=5000, description="Server port")
    
    # Database Configuration
    MONGODB_URL: str = Field(default="mongodb+srv://dnyaneshwartanpuremulsanit:haH5719avivl1uzf@cluster0.5yajgad.mongodb.net/mental_health_detection", description="MongoDB connection URL with database name")
    DATABASE_NAME: str = Field(default="mental_health_detection", description="Database name")
    
    # Twitter API Configuration (twitterapi.io)
    # Note: twitterapi.io provides a single API key for simplified access
    TWITTER_API_KEY: str = Field(default="", description="Primary Twitter API key from twitterapi.io")
    
    # Optional additional keys (if provided by twitterapi.io)
    TWITTER_API_SECRET: str = Field(default="", description="Twitter API secret (optional)")
    TWITTER_BEARER_TOKEN: str = Field(default="", description="Twitter Bearer token (optional)")
    
    # twitterapi.io specific settings
    TWITTER_API_BASE_URL: str = Field(default="https://api.twitterapi.io/v1", description="twitterapi.io base URL")
    TWITTER_RATE_LIMIT: int = Field(default=100, description="Requests per 15-minute window")
    TWITTER_TIMEOUT: int = Field(default=30, description="Request timeout in seconds")
    
    # JWT Configuration
    JWT_SECRET_KEY: str = Field(default="your-secret-key", description="JWT secret key")
    JWT_ALGORITHM: str = Field(default="HS256", description="JWT algorithm")
    JWT_EXPIRATION_HOURS: int = Field(default=24, description="JWT expiration in hours")
    
    # CORS Configuration
    ALLOWED_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"],
        description="Allowed CORS origins"
    )
    
    # File Upload Configuration
    MAX_FILE_SIZE: int = Field(default=10485760, description="Maximum file size in bytes (10MB)")
    ALLOWED_FILE_TYPES: List[str] = Field(
        default=["text/csv", "application/csv"],
        description="Allowed file types for upload"
    )
    
    # AI Models Configuration
    MODEL_PATH: str = Field(default="./models/", description="Path to AI models directory")
    CNN_MODEL_PATH: str = Field(default="./models/cnn_model.h5", description="CNN model path")
    DNN_MODEL_PATH: str = Field(default="./models/dnn_model.h5", description="DNN model path")
    CASTLE_MODEL_PATH: str = Field(default="./models/castle_model.pkl", description="CASTLE model path")
    MOON_MODEL_PATH: str = Field(default="./models/moon_model.h5", description="MOON model path")
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = Field(default=60, description="Rate limit per minute")
    RATE_LIMIT_PER_HOUR: int = Field(default=1000, description="Rate limit per hour")
    
    # Logging Configuration
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    LOG_FILE: str = Field(default="./logs/app.log", description="Log file path")
    
    # Data Processing Configuration
    MAX_TEXT_LENGTH: int = Field(default=10000, description="Maximum text length for processing")
    MIN_TEXT_LENGTH: int = Field(default=10, description="Minimum text length for processing")
    
    # Twitter Data Collection
    MAX_TWEETS_PER_USER: int = Field(default=15, description="Maximum tweets to collect per user")
    
    # Remove duplicate TWITTER_TIMEOUT since it's defined above
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True
    )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ensure_directories()
    
    def _ensure_directories(self):
        """
        Ensure required directories exist
        """
        # Create models directory
        model_dir = Path(self.MODEL_PATH)
        model_dir.mkdir(exist_ok=True)
        
        # Create logs directory
        log_dir = Path(self.LOG_FILE).parent
        log_dir.mkdir(exist_ok=True)
    
    @property
    def database_url(self) -> str:
        """
        Get the complete database URL
        """
        return f"{self.MONGODB_URL}/{self.DATABASE_NAME}"
    
    @property
    def is_development(self) -> bool:
        """
        Check if running in development mode
        """
        return self.ENVIRONMENT.lower() == "development"
    
    @property
    def is_production(self) -> bool:
        """
        Check if running in production mode
        """
        return self.ENVIRONMENT.lower() == "production"


# Create global settings instance
settings = Settings()

# Print configuration summary (only in development)
if settings.is_development:
    print("🔧 Configuration loaded:")
    print(f"   Environment: {settings.ENVIRONMENT}")
    print(f"   Database: {settings.DATABASE_NAME}")
    print(f"   Server: {settings.HOST}:{settings.PORT}")
    print(f"   Models Path: {settings.MODEL_PATH}")
    print(f"   Log Level: {settings.LOG_LEVEL}")
