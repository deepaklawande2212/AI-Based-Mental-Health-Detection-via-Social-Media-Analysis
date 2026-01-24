"""
Health Check and System Status API Routes

This module provides system monitoring and health check endpoints.

Endpoints:
- GET /health - Basic health check
- GET /status - Detailed system status
- GET /info - System information
"""

from fastapi import APIRouter, HTTPException
from datetime import datetime
from loguru import logger
import psutil
import os

from app.config.database import get_database


router = APIRouter()


@router.get("/health")
async def health_check():
    """
    Basic health check endpoint
    
    Returns:
        dict: Health status
    """
    try:
        # Check if models are loaded (from app state)
        from app.main import app
        models_loaded = hasattr(app.state, 'model_manager') and app.state.model_manager is not None
        
        return {
            "status": "healthy" if models_loaded else "initializing",
            "timestamp": datetime.utcnow().isoformat(),
            "database": "optional",  # Database is optional for this system
            "models": "loaded" if models_loaded else "loading",
            "message": "Depression Detection System is running"
        }
        
    except Exception as e:
        logger.error(f"❌ Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "database": "optional",
            "models": "error",
            "error": str(e)
        }


@router.get("/status")
async def system_status():
    """
    Detailed system status including database and models
    
    Returns:
        dict: Comprehensive system status
    """
    try:
        logger.info("🔍 API request for system status")
        
        status = {
            "timestamp": datetime.utcnow().isoformat(),
            "system": {},
            "database": {},
            "models": {},
            "services": {}
        }
        
        # System information
        try:
            uptime = datetime.utcnow() - datetime.fromtimestamp(psutil.boot_time())
            status["system"] = {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory": {
                    "total": psutil.virtual_memory().total,
                    "available": psutil.virtual_memory().available,
                    "percent": psutil.virtual_memory().percent
                },
                "disk": {
                    "total": psutil.disk_usage('/').total,
                    "free": psutil.disk_usage('/').free,
                    "percent": psutil.disk_usage('/').percent
                },
                "uptime": uptime.total_seconds()
            }
        except Exception as e:
            status["system"] = {"error": f"Could not get system info: {str(e)}"}
        
        # Database status (optional)
        try:
            db = await get_database()
            await db.command("ping")
            
            # Get collection stats
            from app.config.database import get_twitter_collection, get_csv_collection, get_analysis_collection
            
            twitter_collection = await get_twitter_collection()
            csv_collection = await get_csv_collection()
            analysis_collection = await get_analysis_collection()
            
            twitter_count = await twitter_collection.count_documents({})
            csv_count = await csv_collection.count_documents({})
            analysis_count = await analysis_collection.count_documents({})
            
            status["database"] = {
                "status": "connected",
                "collections": {
                    "twitter_data": twitter_count,
                    "csv_data": csv_count,
                    "analysis_results": analysis_count
                }
            }
        except Exception as e:
            status["database"] = {
                "status": "optional",
                "message": "Database is optional for this system",
                "error": str(e)
            }
        
        # AI Models status
        try:
            from app.main import app
            if hasattr(app.state, 'model_manager'):
                model_manager = app.state.model_manager
                
                models = {
                    "decision_tree": "decision_tree" in model_manager.models and model_manager.models["decision_tree"] is not None,
                    "cnn": "cnn" in model_manager.models and model_manager.models["cnn"] is not None,
                    "lstm": "lstm" in model_manager.models and model_manager.models["lstm"] is not None,
                    "rnn": "rnn" in model_manager.models and model_manager.models["rnn"] is not None,
                    "bert": "bert" in model_manager.models and model_manager.models["bert"] is not None
                }
                
                loaded_count = sum(models.values())
                
                status["models"] = {
                    "status": "available" if loaded_count > 0 else "unavailable",
                    "loaded_models": loaded_count,
                    "total_models": 5,
                    "details": models,
                    "accuracies": model_manager.model_accuracies if hasattr(model_manager, 'model_accuracies') else {}
                }
            else:
                status["models"] = {
                    "status": "not_initialized",
                    "loaded_models": 0,
                    "total_models": 5
                }
        except Exception as e:
            status["models"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Services status
        status["services"] = {
            "twitter_api": "configured",  # Would check API key validity in production
            "preprocessing": "available",
            "ai_models": "available" if status["models"]["status"] == "available" else "unavailable",
            "csv_processing": "available"
        }
        
        return status
        
    except Exception as e:
        logger.error(f"❌ Error getting system status: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/info")
async def system_info():
    """
    Basic system information
    
    Returns:
        dict: System information
    """
    try:
        return {
            "name": "Depression Detection System",
            "version": "1.0.0",
            "description": "AI-powered mental health analysis platform",
            "framework": "FastAPI",
            "python_version": os.sys.version,
            "features": [
                "Twitter data collection",
                "CSV file processing",
                "AI-powered sentiment analysis",
                "Emotion detection",
                "Risk assessment",
                "Multiple ML models (CNN, DNN, CASTLE, MOON)"
            ],
            "endpoints": {
                "twitter": "/api/twitter",
                "csv": "/api/csv",
                "analysis": "/api/analysis",
                "health": "/api/health"
            },
            "docs": "/docs",
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting system info: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/metrics")
async def system_metrics():
    """
    System performance metrics
    
    Returns:
        dict: Performance metrics
    """
    try:
        import time
        from datetime import timedelta
        
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Process information
        process = psutil.Process()
        process_info = {
            "pid": process.pid,
            "memory_mb": process.memory_info().rss / 1024 / 1024,
            "cpu_percent": process.cpu_percent(),
            "create_time": datetime.fromtimestamp(process.create_time()),
            "num_threads": process.num_threads()
        }
        
        # Database metrics
        db_metrics = {"status": "unknown"}
        try:
            from app.config.database import get_analysis_collection
            collection = await get_analysis_collection()
            
            # Recent analysis counts
            from datetime import datetime, timedelta
            
            now = datetime.utcnow()
            last_hour = now - timedelta(hours=1)
            last_day = now - timedelta(days=1)
            
            recent_analyses = await collection.count_documents({"created_at": {"$gte": last_hour}})
            daily_analyses = await collection.count_documents({"created_at": {"$gte": last_day}})
            
            db_metrics = {
                "status": "connected",
                "analyses_last_hour": recent_analyses,
                "analyses_last_day": daily_analyses
            }
            
        except Exception as e:
            db_metrics = {"status": "error", "error": str(e)}
        
        return {
            "timestamp": datetime.utcnow(),
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": disk.percent,
                "memory_available_gb": round(memory.available / 1024 / 1024 / 1024, 2),
                "disk_free_gb": round(disk.free / 1024 / 1024 / 1024, 2)
            },
            "process": process_info,
            "database": db_metrics
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting system metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
