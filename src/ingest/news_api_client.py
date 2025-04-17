"""
NewsAPI client for fetching financial news and publishing to Kafka.
"""
import os
import json
import time
import logging
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
from kafka import KafkaProducer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
KAFKA_BOOTSTRAP = os.getenv('KAFKA_BOOTSTRAP', 'localhost:9092')
NEWS_API_KEY = os.getenv('NEWS_API_KEY', '')
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 500))

class NewsAPIClient:
    """Client for fetching financial news from NewsAPI and publishing to Kafka."""
    
    def __init__(self, api_key=None, kafka_bootstrap=None):
        """Initialize the NewsAPI client.
        
        Args:
            api_key (str, optional): NewsAPI key. Defaults to environment variable.
            kafka_bootstrap (str, optional): Kafka bootstrap server. Defaults to environment variable.
        """
        self.api_key = api_key or NEWS_API_KEY
        if not self.api_key:
            raise ValueError("NewsAPI key is required")
        
        self.base_url = "https://newsapi.org/v2"
        self.kafka_bootstrap = kafka_bootstrap or KAFKA_BOOTSTRAP
        
        # Initialize Kafka producer
        self.producer = KafkaProducer(
            bootstrap_servers=self.kafka_bootstrap,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        logger.info(f"Initialized NewsAPI client with Kafka bootstrap: {self.kafka_bootstrap}")
    
    def fetch_financial_news(self, query="finance OR stock OR market", days_back=1, 
                            language="en", sort_by="publishedAt", page_size=100):
        """Fetch financial news articles.
        
        Args:
            query (str, optional): Search query. Defaults to "finance OR stock OR market".
            days_back (int, optional): Number of days to look back. Defaults to 1.
            language (str, optional): Article language. Defaults to "en".
            sort_by (str, optional): Sort order. Defaults to "publishedAt".
            page_size (int, optional): Number of results per page. Defaults to 100.
            
        Returns:
            list: News articles.
        """
        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        url = f"{self.base_url}/everything"
        params = {
            'q': query,
            'from': from_date,
            'language': language,
            'sortBy': sort_by,
            'pageSize': min(page_size, BATCH_SIZE),
            'apiKey': self.api_key
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data.get('status') == 'ok' and 'articles' in data:
                return data['articles']
            else:
                logger.warning(f"No articles found or API error: {data.get('message', 'Unknown error')}")
                return []
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching news: {e}")
            return []
    
    def publish_to_kafka(self, articles):
        """Publish news articles to Kafka.
        
        Args:
            articles (list): The articles to publish.
            
        Returns:
            int: Number of articles published.
        """
        count = 0
        for article in articles:
            # Map fields to our schema
            record = {
                'timestamp': article.get('publishedAt'),
                'title': article.get('title'),
                'description': article.get('description'),
                'content': article.get('content'),
                'source': article.get('source', {}).get('name'),
                'url': article.get('url'),
                'author': article.get('author')
            }
            
            # Publish to Kafka
            self.producer.send('news', record)
            count += 1
        
        # Ensure all messages are sent
        self.producer.flush()
        logger.info(f"Published {count} news articles to Kafka")
        return count
    
    def fetch_and_publish(self, queries=None, days_back=1):
        """Fetch news for multiple queries and publish to Kafka.
        
        Args:
            queries (list, optional): List of search queries. Defaults to financial topics.
            days_back (int, optional): Number of days to look back. Defaults to 1.
            
        Returns:
            dict: Summary of articles published per query.
        """
        if queries is None:
            queries = [
                "finance OR stock OR market",
                "economy OR economic",
                "cryptocurrency OR bitcoin",
                "investment OR investing",
                "trading OR trader"
            ]
        
        results = {}
        for query in queries:
            logger.info(f"Fetching news for query: {query}")
            articles = self.fetch_financial_news(query=query, days_back=days_back)
            if articles:
                count = self.publish_to_kafka(articles)
                results[query] = count
            else:
                results[query] = 0
                
            # Avoid rate limiting
            time.sleep(1)
        
        return results

def main():
    """Main function to demonstrate usage."""
    # Check if API key is set
    if not NEWS_API_KEY:
        logger.error("Please set the NEWS_API_KEY environment variable")
        return
    
    # Initialize client
    client = NewsAPIClient()
    
    # Example usage
    results = client.fetch_and_publish(days_back=7)
    
    # Print summary
    logger.info("Summary of published articles:")
    for query, count in results.items():
        logger.info(f"{query}: {count} articles")

if __name__ == "__main__":
    main()
