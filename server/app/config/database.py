"""
Database Configuration and Connection

This module handles MongoDB connection and database operations.
It provides async connection management and database utilities.

Phase 1: MongoDB Setup
- Connection management
- Database initialization
- Collection setup
"""

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import ServerSelectionTimeoutError
from loguru import logger
from typing import Optional
import certifi

from app.config.settings import settings

# Global database connection
_client: Optional[AsyncIOMotorClient] = None
_database: Optional[AsyncIOMotorDatabase] = None


async def connect_to_mongo():
    """
    Create database connection and initialize collections
    """
    global _client, _database
    
    try:
        logger.info(f"🔌 Connecting to MongoDB: {settings.MONGODB_URL}")
        
        # Create MongoDB client
        _client = AsyncIOMotorClient(
            settings.MONGODB_URL,
            serverSelectionTimeoutMS=10000,  # 10 second timeout
            maxPoolSize=10,
            minPoolSize=1,
            tlsCAFile=certifi.where(),  # Use certifi for SSL certificates
            retryWrites=True,
            w="majority"
        )
        
        # Test connection
        await _client.admin.command('ping')
        logger.info("✅ MongoDB connection successful")
        
        # Get database
        _database = _client[settings.DATABASE_NAME]
        
        # Initialize collections
        await initialize_collections()
        
    except ServerSelectionTimeoutError:
        logger.error("❌ Failed to connect to MongoDB - Server selection timeout")
        raise Exception("Could not connect to MongoDB")
    except Exception as e:
        logger.error(f"❌ Database connection error: {str(e)}")
        raise Exception(f"Database connection failed: {str(e)}")


async def close_mongo_connection():
    """
    Close database connection
    """
    global _client
    
    if _client:
        _client.close()
        logger.info("🔐 MongoDB connection closed")


async def get_database() -> AsyncIOMotorDatabase:
    """
    Get database instance
    """
    if _database is None:
        raise Exception("Database not initialized. Call connect_to_mongo() first.")
    return _database


async def initialize_collections():
    """
    Initialize database collections with proper indexes
    
    Collections:
    1. twitter_data - Stores Twitter user data and tweets
    2. csv_data - Stores uploaded CSV files and data
    3. analysis_results - Stores analysis results
    4. text_analyses - Stores text analysis results with model accuracy
    5. twitter_analyses - Stores Twitter analysis results with model accuracy
    6. users - Stores user information (if needed)
    """
    try:
        db = await get_database()
        
        # Get existing collections
        existing_collections = await db.list_collection_names()
        logger.info(f"📁 Existing collections: {existing_collections}")
        
        # Initialize twitter_data collection
        if "twitter_data" not in existing_collections:
            await db.create_collection("twitter_data")
            logger.info("📊 Created twitter_data collection")
        
        # Create indexes for twitter_data
        await db.twitter_data.create_index("username", unique=True)
        await db.twitter_data.create_index("created_at")
        await db.twitter_data.create_index("last_updated")
        
        # Initialize csv_data collection
        if "csv_data" not in existing_collections:
            await db.create_collection("csv_data")
            logger.info("📊 Created csv_data collection")
        
        # Create indexes for csv_data
        await db.csv_data.create_index("file_id", unique=True)
        await db.csv_data.create_index("upload_date")
        await db.csv_data.create_index("filename")
        
        # Initialize analysis_results collection
        if "analysis_results" not in existing_collections:
            await db.create_collection("analysis_results")
            logger.info("📊 Created analysis_results collection")
        
        # Create indexes for analysis_results
        await db.analysis_results.create_index("analysis_id", unique=True)
        await db.analysis_results.create_index("analysis_type")
        await db.analysis_results.create_index("created_at")
        await db.analysis_results.create_index("username")  # For Twitter analysis
        await db.analysis_results.create_index("file_id")   # For CSV analysis
        
        # Initialize text_analyses collection
        if "text_analyses" not in existing_collections:
            await db.create_collection("text_analyses")
            logger.info("📊 Created text_analyses collection")
        
        # Create indexes for text_analyses
        await db.text_analyses.create_index("analysis_id", unique=True)
        await db.text_analyses.create_index("created_at")
        await db.text_analyses.create_index("best_model")
        await db.text_analyses.create_index("overall_accuracy")
        
        # Initialize twitter_analyses collection
        if "twitter_analyses" not in existing_collections:
            await db.create_collection("twitter_analyses")
            logger.info("📊 Created twitter_analyses collection")
        
        # Create indexes for twitter_analyses
        await db.twitter_analyses.create_index("analysis_id", unique=True)
        await db.twitter_analyses.create_index("username")
        await db.twitter_analyses.create_index("created_at")
        await db.twitter_analyses.create_index("best_model")
        await db.twitter_analyses.create_index("overall_accuracy")
        
        logger.info("✅ Database collections initialized successfully")
        
        # Print collection statistics
        twitter_count = await db.twitter_data.count_documents({})
        csv_count = await db.csv_data.count_documents({})
        analysis_count = await db.analysis_results.count_documents({})
        text_analyses_count = await db.text_analyses.count_documents({})
        twitter_analyses_count = await db.twitter_analyses.count_documents({})
        
        logger.info(f"📈 Collection Statistics:")
        logger.info(f"   Twitter Data: {twitter_count} records")
        logger.info(f"   CSV Data: {csv_count} files")
        logger.info(f"   Analysis Results: {analysis_count} analyses")
        logger.info(f"   Text Analyses: {text_analyses_count} analyses")
        logger.info(f"   Twitter Analyses: {twitter_analyses_count} analyses")
        
    except Exception as e:
        logger.error(f"❌ Error initializing collections: {str(e)}")
        raise Exception(f"Failed to initialize database collections: {str(e)}")


# Collection helper functions
async def get_twitter_collection():
    """Get twitter_data collection"""
    db = await get_database()
    return db.twitter_data


async def get_csv_collection():
    """Get csv_data collection"""
    db = await get_database()
    return db.csv_data


async def get_analysis_collection():
    """Get analysis_results collection"""
    db = await get_database()
    return db.analysis_results


async def get_text_analyses_collection():
    """Get text_analyses collection"""
    db = await get_database()
    return db.text_analyses


async def get_twitter_analyses_collection():
    """Get twitter_analyses collection"""
    db = await get_database()
    return db.twitter_analyses


async def health_check():
    """
    Check database health
    """
    try:
        db = await get_database()
        
        # Test connection with a simple operation
        result = await db.command("ping")
        
        # Get database stats
        stats = await db.command("dbstats")
        
        return {
            "status": "healthy",
            "connection": "active",
            "database": settings.DATABASE_NAME,
            "collections_count": stats.get("collections", 0),
            "data_size": stats.get("dataSize", 0),
            "storage_size": stats.get("storageSize", 0)
        }
    except Exception as e:
        logger.error(f"❌ Database health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }
