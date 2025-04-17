"""
Twitter API client for streaming tweets and publishing to Kafka.
"""
import os
import json
import logging
import time
from datetime import datetime
from dotenv import load_dotenv
from kafka import KafkaProducer
import tweepy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
KAFKA_BOOTSTRAP = os.getenv('KAFKA_BOOTSTRAP', 'localhost:9092')
TWITTER_API_KEY = os.getenv('TWITTER_API_KEY', '')
TWITTER_API_SECRET = os.getenv('TWITTER_API_SECRET', '')
TWITTER_ACCESS_TOKEN = os.getenv('TWITTER_ACCESS_TOKEN', '')
TWITTER_ACCESS_SECRET = os.getenv('TWITTER_ACCESS_SECRET', '')
TWITTER_BEARER_TOKEN = os.getenv('TWITTER_BEARER_TOKEN', '')

class TweetListener(tweepy.StreamingClient):
    """Listener for Twitter stream that publishes tweets to Kafka."""
    
    def __init__(self, bearer_token, kafka_bootstrap=None):
        """Initialize the tweet listener.
        
        Args:
            bearer_token (str): Twitter API bearer token.
            kafka_bootstrap (str, optional): Kafka bootstrap server. Defaults to environment variable.
        """
        super().__init__(bearer_token)
        self.kafka_bootstrap = kafka_bootstrap or KAFKA_BOOTSTRAP
        
        # Initialize Kafka producer
        self.producer = KafkaProducer(
            bootstrap_servers=self.kafka_bootstrap,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        logger.info(f"Initialized Tweet listener with Kafka bootstrap: {self.kafka_bootstrap}")
        
        # Track tweet count
        self.tweet_count = 0
    
    def on_tweet(self, tweet):
        """Process a tweet and publish to Kafka.
        
        Args:
            tweet: The tweet object from the Twitter API.
        """
        # Extract relevant fields
        record = {
            'id': tweet.id,
            'timestamp': tweet.created_at.isoformat() if hasattr(tweet, 'created_at') else datetime.now().isoformat(),
            'text': tweet.text,
            'user': tweet.author_id,
            'retweet_count': tweet.public_metrics.get('retweet_count', 0) if hasattr(tweet, 'public_metrics') else 0,
            'like_count': tweet.public_metrics.get('like_count', 0) if hasattr(tweet, 'public_metrics') else 0,
            'reply_count': tweet.public_metrics.get('reply_count', 0) if hasattr(tweet, 'public_metrics') else 0
        }
        
        # Publish to Kafka
        self.producer.send('tweets', record)
        self.tweet_count += 1
        
        # Log progress periodically
        if self.tweet_count % 100 == 0:
            logger.info(f"Processed {self.tweet_count} tweets")
    
    def on_error(self, status):
        """Handle stream errors.
        
        Args:
            status: The error status code.
        """
        logger.error(f"Twitter stream error: {status}")
        if status == 420:  # Rate limit
            logger.warning("Rate limited, sleeping for 1 minute")
            time.sleep(60)
            return True  # Continue streaming
        return True  # Continue streaming

class TweetStreamer:
    """Client for streaming tweets from Twitter API and publishing to Kafka."""
    
    def __init__(self, bearer_token=None, kafka_bootstrap=None):
        """Initialize the tweet streamer.
        
        Args:
            bearer_token (str, optional): Twitter API bearer token. Defaults to environment variable.
            kafka_bootstrap (str, optional): Kafka bootstrap server. Defaults to environment variable.
        """
        self.bearer_token = bearer_token or TWITTER_BEARER_TOKEN
        if not self.bearer_token:
            raise ValueError("Twitter bearer token is required")
        
        self.kafka_bootstrap = kafka_bootstrap or KAFKA_BOOTSTRAP
        self.listener = TweetListener(self.bearer_token, self.kafka_bootstrap)
    
    def add_rules(self, terms):
        """Add filter rules to the stream.
        
        Args:
            terms (list): List of search terms or hashtags.
            
        Returns:
            list: Added rules.
        """
        # First delete existing rules
        rules = self.listener.get_rules()
        if rules.data:
            rule_ids = [rule.id for rule in rules.data]
            self.listener.delete_rules(rule_ids)
        
        # Add new rules
        rules = []
        for term in terms:
            rules.append(tweepy.StreamRule(term))
        
        response = self.listener.add_rules(rules)
        logger.info(f"Added {len(response.data)} stream rules")
        return response.data
    
    def start_stream(self, terms=None):
        """Start streaming tweets.
        
        Args:
            terms (list, optional): List of search terms or hashtags. Defaults to financial terms.
        """
        if terms is None:
            terms = [
                "stock market",
                "#investing",
                "#trading",
                "#finance",
                "#stocks",
                "#cryptocurrency",
                "#bitcoin",
                "#ethereum",
                "financial news",
                "market analysis"
            ]
        
        # Add rules
        self.add_rules(terms)
        
        # Start stream
        logger.info("Starting Twitter stream...")
        self.listener.filter(
            tweet_fields=['created_at', 'public_metrics', 'author_id'],
            expansions=['author_id']
        )

def main():
    """Main function to demonstrate usage."""
    # Check if API keys are set
    if not TWITTER_BEARER_TOKEN:
        logger.error("Please set the TWITTER_BEARER_TOKEN environment variable")
        return
    
    # Initialize streamer
    streamer = TweetStreamer()
    
    # Start streaming
    try:
        streamer.start_stream()
    except KeyboardInterrupt:
        logger.info("Stream stopped by user")
    except Exception as e:
        logger.error(f"Stream error: {e}")

if __name__ == "__main__":
    main()
