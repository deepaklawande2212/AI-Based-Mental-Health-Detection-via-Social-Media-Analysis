"""
Twitter Data Collection Service

This service handles Twitter data collection using twitterapi.io.
It collects user information and tweets for mental health analysis.

Phase 1: Data Collection from Twitter
- User profile data collection
- Tweet collection and storage
- Rate limiting and error handling
- Data validation and cleaning
"""

import httpx
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from loguru import logger

from app.config.settings import settings
from app.config.database import get_twitter_collection
from app.models.database_models import TwitterData, Tweet


class TwitterService:
    """
    Service for collecting Twitter data using twitterapi.io
    
    This service provides methods to:
    1. Collect user profile information
    2. Fetch recent tweets from a user
    3. Store data in MongoDB
    4. Handle rate limiting and errors
    """
    
    def __init__(self):
        self.base_url = "https://api.twitterapi.io/v1"
        self.api_key = settings.TWITTER_API_KEY
        self.timeout = settings.TWITTER_TIMEOUT
        
        # HTTP client with proper headers
        self.client = httpx.AsyncClient(
            timeout=self.timeout,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        )
    
    async def collect_user_data(self, username: str, max_tweets: int = 100) -> TwitterData:
        """
        Collect complete user data including profile and tweets
        
        Args:
            username: Twitter username (without @)
            max_tweets: Maximum number of tweets to collect
            
        Returns:
            TwitterData: Complete user data with tweets
        """
        try:
            logger.info(f"🐦 Starting data collection for @{username}")
            
            # Check if data already exists
            existing_data = await self._get_existing_data(username)
            if existing_data and self._is_data_fresh(existing_data):
                logger.info(f"✅ Using cached data for @{username}")
                return existing_data
            
            # Collect user profile
            user_profile = await self._get_user_profile(username)
            if not user_profile:
                raise Exception(f"User @{username} not found or profile is private")
            
            # Collect tweets
            tweets = await self._get_user_tweets(username, max_tweets)
            
            # Create TwitterData object
            twitter_data = TwitterData(
                username=username,
                user_id=user_profile.get("id"),
                display_name=user_profile.get("name"),
                bio=user_profile.get("description", ""),
                followers_count=user_profile.get("public_metrics", {}).get("followers_count", 0),
                following_count=user_profile.get("public_metrics", {}).get("following_count", 0),
                tweet_count=user_profile.get("public_metrics", {}).get("tweet_count", 0),
                account_created=self._parse_twitter_date(user_profile.get("created_at")),
                tweets=tweets,
                total_tweets_collected=len(tweets),
                collection_status="completed"
            )
            
            # Store in database
            await self._store_twitter_data(twitter_data)
            
            logger.info(f"✅ Successfully collected {len(tweets)} tweets for @{username}")
            return twitter_data
            
        except Exception as e:
            logger.error(f"❌ Error collecting data for @{username}: {str(e)}")
            
            # Store error information
            error_data = TwitterData(
                username=username,
                collection_status="failed",
                error_message=str(e)
            )
            await self._store_twitter_data(error_data)
            
            raise Exception(f"Failed to collect Twitter data: {str(e)}")
    
    async def _get_user_profile(self, username: str) -> Optional[Dict[str, Any]]:
        """
        Get user profile information from Twitter API
        
        Args:
            username: Twitter username
            
        Returns:
            Dict: User profile data or None if not found
        """
        try:
            url = f"{self.base_url}/users/by/username/{username}"
            params = {
                "user.fields": "created_at,description,public_metrics,verified"
            }
            
            response = await self.client.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                if "data" in data:
                    logger.info(f"✅ User profile collected for @{username}")
                    return data["data"]
                else:
                    logger.warning(f"⚠️ No user data found for @{username}")
                    return None
            
            elif response.status_code == 404:
                logger.warning(f"⚠️ User @{username} not found")
                return None
            
            elif response.status_code == 429:
                logger.warning("⚠️ Rate limit exceeded, waiting...")
                await asyncio.sleep(60)  # Wait 1 minute
                return await self._get_user_profile(username)  # Retry
            
            else:
                logger.error(f"❌ Twitter API error: {response.status_code} - {response.text}")
                return None
                
        except httpx.TimeoutException:
            logger.error(f"❌ Timeout while fetching profile for @{username}")
            return None
        except Exception as e:
            logger.error(f"❌ Error fetching user profile: {str(e)}")
            return None
    
    async def _get_user_tweets(self, username: str, max_tweets: int) -> List[Tweet]:
        """
        Get recent tweets from a user
        
        Args:
            username: Twitter username
            max_tweets: Maximum number of tweets to collect
            
        Returns:
            List[Tweet]: List of tweets
        """
        try:
            url = f"{self.base_url}/users/by/username/{username}/tweets"
            params = {
                "max_results": min(max_tweets, 100),  # API limit is 100 per request
                "tweet.fields": "created_at,public_metrics,lang,context_annotations",
                "exclude": "retweets,replies"  # Exclude retweets and replies for cleaner data
            }
            
            all_tweets = []
            pagination_token = None
            
            while len(all_tweets) < max_tweets:
                if pagination_token:
                    params["pagination_token"] = pagination_token
                
                response = await self.client.get(url, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if "data" in data:
                        tweets_data = data["data"]
                        
                        for tweet_data in tweets_data:
                            if len(all_tweets) >= max_tweets:
                                break
                            
                            tweet = Tweet(
                                id=tweet_data["id"],
                                text=tweet_data["text"],
                                created_at=self._parse_twitter_date(tweet_data["created_at"]),
                                retweet_count=tweet_data.get("public_metrics", {}).get("retweet_count", 0),
                                like_count=tweet_data.get("public_metrics", {}).get("like_count", 0),
                                reply_count=tweet_data.get("public_metrics", {}).get("reply_count", 0),
                                language=tweet_data.get("lang", "en"),
                                is_retweet=tweet_data["text"].startswith("RT @")
                            )
                            all_tweets.append(tweet)
                        
                        # Check for pagination
                        if "meta" in data and "next_token" in data["meta"]:
                            pagination_token = data["meta"]["next_token"]
                        else:
                            break
                    else:
                        logger.warning(f"⚠️ No tweets found for @{username}")
                        break
                
                elif response.status_code == 429:
                    logger.warning("⚠️ Rate limit exceeded, waiting...")
                    await asyncio.sleep(60)
                    continue
                
                else:
                    logger.error(f"❌ Error fetching tweets: {response.status_code} - {response.text}")
                    break
            
            logger.info(f"✅ Collected {len(all_tweets)} tweets for @{username}")
            return all_tweets
            
        except Exception as e:
            logger.error(f"❌ Error fetching tweets: {str(e)}")
            return []
    
    async def _get_existing_data(self, username: str) -> Optional[TwitterData]:
        """
        Check if we already have data for this user
        
        Args:
            username: Twitter username
            
        Returns:
            TwitterData or None: Existing data if found
        """
        try:
            collection = await get_twitter_collection()
            doc = await collection.find_one({"username": username})
            
            if doc:
                return TwitterData(**doc)
            return None
            
        except Exception as e:
            logger.error(f"❌ Error checking existing data: {str(e)}")
            return None
    
    def _is_data_fresh(self, twitter_data: TwitterData, hours: int = 24) -> bool:
        """
        Check if existing data is still fresh (not older than specified hours)
        
        Args:
            twitter_data: Existing Twitter data
            hours: Maximum age in hours
            
        Returns:
            bool: True if data is fresh
        """
        if not twitter_data.last_updated:
            return False
        
        age = datetime.utcnow() - twitter_data.last_updated
        return age < timedelta(hours=hours)
    
    async def _store_twitter_data(self, twitter_data: TwitterData):
        """
        Store or update Twitter data in MongoDB
        
        Args:
            twitter_data: TwitterData object to store
        """
        try:
            collection = await get_twitter_collection()
            
            # Convert to dict and handle datetime serialization
            data_dict = twitter_data.dict(exclude={"id"})
            data_dict["last_updated"] = datetime.utcnow()
            
            # Upsert (update if exists, insert if not)
            result = await collection.replace_one(
                {"username": twitter_data.username},
                data_dict,
                upsert=True
            )
            
            if result.upserted_id:
                logger.info(f"📝 Stored new Twitter data for @{twitter_data.username}")
            else:
                logger.info(f"📝 Updated Twitter data for @{twitter_data.username}")
                
        except Exception as e:
            logger.error(f"❌ Error storing Twitter data: {str(e)}")
            raise
    
    def _parse_twitter_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """
        Parse Twitter date string to datetime object
        
        Args:
            date_str: Twitter date string
            
        Returns:
            datetime or None: Parsed datetime
        """
        if not date_str:
            return None
        
        try:
            # Twitter uses ISO format: 2023-01-01T12:00:00.000Z
            return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        except Exception as e:
            logger.error(f"❌ Error parsing date '{date_str}': {str(e)}")
            return None
    
    async def get_stored_data(self, username: str) -> Optional[TwitterData]:
        """
        Get stored Twitter data for a user
        
        Args:
            username: Twitter username
            
        Returns:
            TwitterData or None: Stored data if found
        """
        return await self._get_existing_data(username)
    
    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()


# Example usage and testing functions
async def test_twitter_service():
    """
    Test function for Twitter service (for development only)
    """
    service = TwitterService()
    
    try:
        # Test with a known public account
        data = await service.collect_user_data("elonmusk", max_tweets=10)
        print(f"Collected data for @{data.username}")
        print(f"Tweets: {len(data.tweets)}")
        print(f"Followers: {data.followers_count}")
        
    except Exception as e:
        print(f"Test failed: {str(e)}")
    
    finally:
        await service.close()


if __name__ == "__main__":
    # Run test if this file is executed directly
    asyncio.run(test_twitter_service())
