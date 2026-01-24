"""
Main FastAPI Application

This is the entry point for the Depression Detection System backend.
It initializes the FastAPI app, sets up middleware, configures routes,
and handles startup/shutdown events.

Features:
- CORS middleware for frontend communication
- Route organization with API prefixes
- MongoDB connection management (optional)
- AI model initialization
- Comprehensive error handling
- Request logging
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
from loguru import logger
import sys
from datetime import datetime

# Import route modules
from app.routes import twitter, csv, analysis, health, enhanced_analysis

# Import services and config
from app.services.ai_models import ModelManager
from app.config.database import connect_to_mongo, close_mongo_connection


# Configure logging
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)
logger.add(
    "logs/app_{time:YYYY-MM-DD}.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG",
    rotation="1 day",
    retention="30 days"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager
    Handles startup and shutdown events
    """
    # Startup
    logger.info("🚀 Starting Depression Detection System...")
    
    try:
        # Initialize database (optional - skip if not available)
        logger.info("📊 Initializing database connection...")
        try:
            await connect_to_mongo()
            logger.success("✅ Database connected successfully")
            app.state.database_available = True
        except Exception as e:
            logger.warning(f"⚠️ Database not available: {str(e)}")
            logger.info("📝 Running without database - some features may be limited")
            app.state.database_available = False
        
        # Initialize AI models
        logger.info("🤖 Initializing AI models...")
        model_manager = ModelManager()
        await model_manager.load_models()
        
        # Store model manager in app state
        app.state.model_manager = model_manager
        logger.success("✅ AI models initialized successfully")
        
        logger.success("🎉 Depression Detection System started successfully!")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Failed to start application: {str(e)}")
        raise
    
    finally:
        # Shutdown
        logger.info("🛑 Shutting down Depression Detection System...")
        
        try:
            # Close database connections if available
            if hasattr(app.state, 'database_available') and app.state.database_available:
                await close_mongo_connection()
                logger.info("📊 Database connections closed")
            
            # Cleanup model manager
            if hasattr(app.state, 'model_manager'):
                delattr(app.state, 'model_manager')
                logger.info("🤖 AI models cleaned up")
            
            logger.success("✅ Application shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ Error during shutdown: {str(e)}")


# Create FastAPI application
app = FastAPI(
    title="Depression Detection System API",
    description="""
    🧠 **Depression Detection System Backend**
    
    A comprehensive AI-powered platform for mental health analysis using social media data and text processing.
    
    ## Features
    
    * **🐦 Twitter Data Collection**: Collect and analyze Twitter data using the Twitter API
    * **📄 CSV Processing**: Upload and analyze text data from CSV files
    * **🤖 AI Models**: Multiple AI models including CNN, DNN, CASTLE, and MOON
    * **📊 Sentiment Analysis**: Positive, negative, and neutral sentiment detection
    * **😊 Emotion Detection**: Joy, sadness, anger, fear, surprise, and disgust analysis
    * **⚠️ Risk Assessment**: Low, medium, and high risk level evaluation
    * **📈 Analytics**: Comprehensive analysis reports and history
    
    ## API Endpoints
    
    * **Twitter**: `/api/twitter` - Twitter data collection and analysis
    * **CSV**: `/api/csv` - CSV file upload and processing
    * **Analysis**: `/api/analysis` - Analysis results and history
    * **Health**: `/api/health` - System health and monitoring
    
    ## Models
    
    * **CNN (86% accuracy)**: Convolutional Neural Network for text classification
    * **DNN (90% accuracy)**: Deep Neural Network for sentiment analysis
    * **CASTLE (95% accuracy)**: Advanced ensemble method for mental health detection
    * **MOON (98% accuracy)**: State-of-the-art model for psychological analysis
    """,
    version="1.0.0",
    contact={
        "name": "Depression Detection System",
        "email": "support@depression-detection.com",
    },
    license_info={
        "name": "MIT",
    },
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React development server
        "http://localhost:3002",  # Vite development server (current)
        "http://localhost:5173",  # Vite development server (default)
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3002",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests"""
    start_time = datetime.now()
    
    # Log request
    logger.info(f"🌐 {request.method} {request.url.path} - {request.client.host}")
    
    try:
        response = await call_next(request)
        
        # Calculate processing time
        process_time = datetime.now() - start_time
        process_time_ms = process_time.total_seconds() * 1000
        
        # Log response
        logger.info(f"✅ {request.method} {request.url.path} - {response.status_code} - {process_time_ms:.2f}ms")
        
        # Add processing time to response headers
        response.headers["X-Process-Time"] = str(process_time_ms)
        
        return response
        
    except Exception as e:
        # Calculate processing time for error case
        process_time = datetime.now() - start_time
        process_time_ms = process_time.total_seconds() * 1000
        
        logger.error(f"❌ {request.method} {request.url.path} - Error - {process_time_ms:.2f}ms - {str(e)}")
        raise


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions"""
    logger.error(f"❌ Unhandled exception: {str(exc)}")
    logger.error(f"Request: {request.method} {request.url}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "path": str(request.url.path)
        }
    )


# HTTP exception handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with proper logging"""
    logger.warning(f"⚠️ HTTP {exc.status_code}: {exc.detail}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat(),
            "path": str(request.url.path)
        }
    )


# Include routers
app.include_router(health.router, prefix="/api/health", tags=["Health"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["Analysis"])
app.include_router(csv.router, prefix="/api/csv", tags=["CSV"])
app.include_router(twitter.router, prefix="/api/twitter", tags=["Twitter"])
app.include_router(enhanced_analysis.router, prefix="/api/enhanced", tags=["Enhanced Analysis"])


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "🧠 Depression Detection System API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/api/health/status"
    }


# Health check endpoint
@app.get("/health")
async def health_check():
    """Simple health check"""
    return {"status": "healthy", "message": "Depression Detection System is running"}


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
