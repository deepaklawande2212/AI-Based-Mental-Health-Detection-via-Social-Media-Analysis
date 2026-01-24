"""
Twitter API Service for twitterapi.io

This service uses twitterapi.io for free Twitter data access.
Uses the advanced search endpoint to collect tweets by username.

Key Features:
- Single API key authentication
- Username-based tweet collection
- Advanced search functionality
- Built-in rate limiting
"""

import asyncio
import aiohttp
import json
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from loguru import logger

from app.config.settings import settings
from app.config.database import get_twitter_collection
from app.models.database_models import TwitterData, Tweet


class TwitterAPIService:
    """
    Twitter service using twitterapi.io
    
    Uses the advanced search endpoint to collect tweets by username
    """
    
    def __init__(self):
        # API key from twitterapi.io
        self.api_key = settings.TWITTER_API_KEY or "54fecf627ad14c1a8b459c7f63300f21"  # Default key
        
        # twitterapi.io endpoints
        self.base_url = "https://api.twitterapi.io"
        self.advanced_search_url = f"{self.base_url}/twitter/tweet/advanced_search"
        
        # Configuration
        self.timeout = 30
        self.max_tweets = settings.MAX_TWEETS_PER_USER
        
        # Session for connection reuse
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Rate limiting
        self.requests_count = 0
        self.last_reset = datetime.now()
        
        # Track seen tweet IDs to avoid duplicates
        self.seen_tweet_ids = set()
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Create HTTP session with authentication"""
        if self.session is None or self.session.closed:
            headers = {
                "x-api-key": self.api_key,
                "Content-Type": "application/json",
                "User-Agent": "Depression-Detection-System/1.0"
            }
            
            self.session = aiohttp.ClientSession(
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        
        return self.session
    
    async def close(self):
        """Close the HTTP session"""
        if self.session and not self.session.closed:
            await self.session.close()
    
    def _check_rate_limit(self):
        """Check rate limiting"""
        now = datetime.now()
        
        # Reset every 15 minutes (more frequent reset)
        if (now - self.last_reset).total_seconds() > 900:  # 15 minutes
            self.requests_count = 0
            self.last_reset = now
        
        # Limit to 50 requests per 15 minutes (more generous)
        if self.requests_count >= 50:
            logger.warning("⏳ Rate limit reached for this 15-minute window")
            return False
        
        return True
    
    def _get_since_time(self, minutes_ago: int = 60) -> int:
        """Get Unix timestamp in seconds for X minutes ago"""
        return int((datetime.now().timestamp() - minutes_ago * 60))
    
    async def collect_user_data(self, username: str, max_tweets: int = 100) -> TwitterData:
        """
        Collect Twitter data for a specific user using advanced search
        
        Args:
            username: Twitter username (without @)
            max_tweets: Maximum number of tweets to collect
            
        Returns:
            TwitterData: Collected user data and tweets
        """
        try:
            logger.info(f"🐦 Collecting Twitter data for @{username}")
            logger.info(f"🐦 API Key: {self.api_key[:10]}...")
            logger.info(f"🐦 Max tweets: {max_tweets}")
            
            if not self._check_rate_limit():
                raise Exception("Rate limit exceeded")
            
            # Clear seen tweet IDs for fresh collection
            self.seen_tweet_ids.clear()
            logger.info(f"🐦 Cleared seen tweet IDs for fresh collection")
            
            # Clean username
            username = username.replace('@', '').strip().lower()
            logger.info(f"🐦 Cleaned username: {username}")
            
            # Collect tweets using advanced search
            tweets, user_info = await self._fetch_user_tweets(username, max_tweets)
            
            logger.info(f"🔍 Collected {len(tweets)} real tweets for @{username}")
            
            if not tweets:
                raise Exception(f"No tweets found for @{username}. The account might be private or have no tweets.")
            
            # Create TwitterData object
            twitter_data = TwitterData(
                user_id=user_info.get('user_id', f"user_{username}"),
                username=username,
                display_name=user_info.get('display_name', username.title()),
                bio=user_info.get('bio', ''),
                followers_count=user_info.get('followers_count', 0),
                following_count=user_info.get('following_count', 0),
                tweet_count=user_info.get('tweet_count', 0),
                tweets=tweets,
                collected_at=datetime.utcnow()
            )
            
            # Try to store in database (skip if not available)
            try:
                collection = await get_twitter_collection()
                await collection.insert_one(twitter_data.dict(exclude={"id"}))
                logger.info(f"✅ Twitter data stored in database: {username}")
            except Exception as e:
                logger.warning(f"⚠️ Could not store Twitter data in database: {str(e)}")
                logger.info("📝 Continuing without database storage")
            
            logger.success(f"✅ Successfully collected {len(tweets)} real tweets for @{username}")
            return twitter_data
        
        except Exception as e:
            logger.error(f"❌ Error collecting Twitter data for @{username}: {str(e)}")
            raise Exception(f"Failed to collect Twitter data: {str(e)}")
    
    async def _fetch_user_tweets(self, username: str, max_tweets: int) -> tuple[List[Tweet], Dict[str, Any]]:
        """
        Fetch tweets for a user using advanced search
        
        Args:
            username: Twitter username
            max_tweets: Maximum tweets to collect
            
        Returns:
            Tuple of (List of Tweet objects, User info dict)
        """
        tweets = []
        user_info = {}
        
        try:
            logger.info(f"🔍 Starting to fetch tweets for @{username}")
            logger.info(f"🔍 Max tweets requested: {max_tweets}")
            session = await self._get_session()
            
            # Get timestamp for 24 hours ago
            since_time = int((datetime.now().timestamp() - 24 * 60 * 60))
            
            # Try different query formats to find tweets
            query_formats = [
                f"from:{username}",
                f"@{username}",
                username
            ]
            
            for query_format in query_formats:
                if len(tweets) >= max_tweets:
                    break
                    
                logger.info(f"🔍 Trying query format: {query_format}")
                
                # Add small delay between requests
                if len(tweets) > 0:
                    import asyncio
                    await asyncio.sleep(0.5)  # 500ms delay
                
                params = {
                    "queryType": "Latest",
                    "query": query_format,
                    "include": "nativeretweets",
                    "since_time": since_time,
                }
                
                logger.info(f"🔍 Making API request with params: {params}")
                logger.info(f"🔍 API URL: {self.advanced_search_url}")
                
                # Make API request
                async with session.get(self.advanced_search_url, params=params) as response:
                    self.requests_count += 1
                    
                    logger.info(f"🔍 API Response Status: {response.status}")
                    
                    if response.status != 200:
                        error_text = await response.text()
                        logger.warning(f"⚠️ Twitter API Error: {response.status} - {error_text}")
                        continue
                    
                    data = await response.json()
                    logger.info(f"🔍 API Response received, data keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
                    
                    api_tweets = data.get('tweets', [])
                    logger.info(f"🔍 Found {len(api_tweets)} tweets in API response")
                    
                    if not api_tweets:
                        logger.warning(f"⚠️ No tweets found for query: {query_format}")
                        continue
                    
                    # Process tweets
                    for i, api_tweet in enumerate(api_tweets):
                        if len(tweets) >= max_tweets:
                            break
                            
                        if not api_tweet.get('id') in self.seen_tweet_ids:
                            logger.info(f"🔍 Processing tweet {i+1}: {api_tweet.get('text', '')[:50]}...")
                            tweet = self._convert_api_tweet_to_tweet(api_tweet)
                            if tweet:
                                tweets.append(tweet)
                                self.seen_tweet_ids.add(api_tweet.get('id'))
                                logger.info(f"✅ Added tweet: {tweet.text[:50]}...")
                                
                                # Extract user info from the first tweet
                                if len(tweets) == 1:
                                    author = api_tweet.get('author', {})
                                    user_info = {
                                        'user_id': author.get('id'),
                                        'display_name': author.get('name'),
                                        'bio': author.get('description', ''),
                                        'followers_count': author.get('followers', 0),
                                        'following_count': author.get('following', 0),
                                        'tweet_count': author.get('statusesCount', 0)
                                    }
                                    logger.info(f"🔍 Extracted user info: {user_info}")
                                # End of if len(tweets) == 1
                            # End of if tweet
                        # End of if not api_tweet.get('id') in self.seen_tweet_ids
                    # End of for i, api_tweet in enumerate(api_tweets)
                # End of for query_format in query_formats
            
            logger.info(f"📊 Collected {len(tweets)} tweets for @{username}")
            return tweets, user_info
        
        except Exception as e:
            logger.error(f"❌ Error fetching tweets for @{username}: {str(e)}")
            logger.error(f"❌ Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"❌ Full traceback: {traceback.format_exc()}")
            return tweets, user_info
    
    def _convert_api_tweet_to_tweet(self, api_tweet: Dict[str, Any]) -> Optional[Tweet]:
        """Convert API tweet format to our Tweet model"""
        try:
            logger.info(f"🔍 Converting API tweet: {api_tweet.get('id')} - {api_tweet.get('text', '')[:30]}...")
            
            # Check if we have the required fields
            if not api_tweet.get('text'):
                logger.warning(f"⚠️ Tweet {api_tweet.get('id')} has no text, skipping")
                return None
            
            # Parse the created_at date
            created_at_str = api_tweet.get('createdAt')
            created_at = None
            if created_at_str:
                try:
                    # Parse Twitter date format: "Wed Aug 13 17:17:33 +0000 2025"
                    from datetime import datetime
                    created_at = datetime.strptime(created_at_str, "%a %b %d %H:%M:%S %z %Y")
                except Exception as e:
                    logger.warning(f"⚠️ Could not parse date '{created_at_str}': {str(e)}")
                    created_at = datetime.utcnow()
            
            tweet = Tweet(
                id=api_tweet.get('id'),  # Use 'id' instead of 'tweet_id'
                text=api_tweet.get('text', ''),
                created_at=created_at,
                retweet_count=api_tweet.get('retweetCount', 0),
                like_count=api_tweet.get('likeCount', 0),
                reply_count=api_tweet.get('replyCount', 0),
                language=api_tweet.get('lang', 'en'),
                is_retweet=api_tweet.get('isReply', False)  # Use isReply as is_retweet
            )
            
            logger.info(f"✅ Successfully converted tweet: {tweet.id} - {tweet.text[:30]}...")
            return tweet
        
        except Exception as e:
            logger.error(f"❌ Error converting tweet: {str(e)}")
            logger.error(f"❌ Tweet data: {api_tweet}")
            return None
    
    def _extract_user_info(self, tweet: Tweet) -> Dict[str, Any]:
        """Extract user information from a tweet"""
        # Since we don't have user info in the Tweet model, we'll need to get it from the API response
        # This method is called with the first tweet, so we can extract user info from there
        return {
            'user_id': None,  # Will be extracted from API response
            'display_name': None,  # Will be extracted from API response
            'bio': '',  # Will be extracted from API response
            'followers_count': 0,  # Will be extracted from API response
            'following_count': 0,  # Will be extracted from API response
            'tweet_count': 0  # Will be extracted from API response
        }


# Global instance
twitter_service = TwitterAPIService()
