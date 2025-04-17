"""
Twitter API v2 client for streaming financial tweets.
"""
import json
import os
import time
from datetime import datetime
from typing import Callable, Dict, List, Optional, Set

import requests
from requests.exceptions import ChunkedEncodingError, ConnectionError, RequestException


class TweetStreamer:
    """Client for streaming tweets from Twitter API v2."""
    
    STREAM_URL = "https://api.twitter.com/2/tweets/search/stream"
    RULES_URL = "https://api.twitter.com/2/tweets/search/stream/rules"
    
    def __init__(self, bearer_token: Optional[str] = None):
        """
        Initialize the Twitter API v2 streamer.
        
        Args:
            bearer_token: Twitter API bearer token. If not provided, will look for TWITTER_BEARER_TOKEN env var.
        """
        self.bearer_token = bearer_token or os.environ.get("TWITTER_BEARER_TOKEN")
        if not self.bearer_token:
            raise ValueError("Twitter bearer token must be provided or set as TWITTER_BEARER_TOKEN env var")
        
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.bearer_token}"})
        
        # Default finance-related keywords and hashtags
        self.default_keywords = [
            "stock market", "stocks", "investing", "finance", "trading", "nasdaq", "nyse", "dow jones",
            "sp500", "s&p 500", "bull market", "bear market", "earnings", "financial", "economy",
            "#stocks", "#investing", "#finance", "#trading", "$AAPL", "$MSFT", "$AMZN", "$GOOGL", "$META"
        ]
    
    def _get_stream_params(self) -> Dict:
        """Get parameters for the filtered stream."""
        return {
            "tweet.fields": "created_at,author_id,text",
            "expansions": "author_id",
            "user.fields": "username"
        }
    
    def _get_rules(self) -> List[Dict]:
        """Get the current stream rules."""
        response = self.session.get(self.RULES_URL)
        response.raise_for_status()
        return response.json().get("data", [])
    
    def _delete_all_rules(self) -> None:
        """Delete all existing stream rules."""
        rules = self._get_rules()
        if not rules:
            return
        
        rule_ids = [rule["id"] for rule in rules]
        payload = {"delete": {"ids": rule_ids}}
        
        response = self.session.post(self.RULES_URL, json=payload)
        response.raise_for_status()
    
    def _set_rules(self, keywords: List[str]) -> None:
        """Set new stream rules based on keywords."""
        # First delete existing rules
        self._delete_all_rules()
        
        # Create rules for keywords (Twitter has a limit on rule complexity)
        rules = []
        for i in range(0, len(keywords), 5):
            batch = keywords[i:i+5]
            rule_value = " OR ".join(batch)
            rules.append({"value": rule_value})
        
        if rules:
            payload = {"add": rules}
            response = self.session.post(self.RULES_URL, json=payload)
            response.raise_for_status()
    
    def start_stream(
        self, 
        callback: Callable[[Dict], None], 
        keywords: Optional[List[str]] = None,
        timeout: int = 90,
        max_retries: int = 3
    ) -> None:
        """
        Start streaming tweets that match the filter rules.
        
        Args:
            callback: Function to call for each tweet received
            keywords: List of keywords/hashtags to filter on (uses default finance keywords if None)
            timeout: Timeout for the streaming connection in seconds
            max_retries: Maximum number of connection retry attempts
        """
        # Set up stream rules
        filter_keywords = keywords or self.default_keywords
        self._set_rules(filter_keywords)
        
        # Start streaming with retries
        retries = 0
        while retries < max_retries:
            try:
                params = self._get_stream_params()
                with self.session.get(
                    self.STREAM_URL, 
                    params=params, 
                    stream=True,
                    timeout=timeout
                ) as response:
                    response.raise_for_status()
                    
                    for line in response.iter_lines():
                        if not line:
                            continue
                        
                        try:
                            tweet_data = json.loads(line)
                            if "data" in tweet_data:
                                tweet = tweet_data["data"]
                                
                                # Extract user information if available
                                username = None
                                if "includes" in tweet_data and "users" in tweet_data["includes"]:
                                    for user in tweet_data["includes"]["users"]:
                                        if user["id"] == tweet["author_id"]:
                                            username = user["username"]
                                            break
                                
                                # Create normalized tweet object
                                normalized_tweet = {
                                    "id": tweet["id"],
                                    "text": tweet["text"],
                                    "created_at": tweet["created_at"],
                                    "author_id": tweet["author_id"],
                                    "username": username,
                                    "timestamp": datetime.now().isoformat()
                                }
                                
                                # Call the callback with the tweet
                                callback(normalized_tweet)
                        
                        except json.JSONDecodeError:
                            continue
                
                # If we get here without exceptions, reset retry counter
                retries = 0
                
            except (ChunkedEncodingError, ConnectionError) as e:
                # These are common for streaming connections, retry with backoff
                retries += 1
                wait_time = 2 ** retries  # Exponential backoff
                print(f"Stream connection error: {str(e)}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                
            except RequestException as e:
                # More serious error
                raise Exception(f"Error connecting to Twitter API: {str(e)}")
    
    def stop_stream(self) -> None:
        """Stop the stream by deleting all rules."""
        self._delete_all_rules()
