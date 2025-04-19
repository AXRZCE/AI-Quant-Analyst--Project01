"""
Script to run the data ingestion pipeline.

This script demonstrates how to use the data pipeline to ingest
stock data, fundamentals, and news articles.
"""

import os
import sys
import argparse
import logging
from datetime import datetime, timedelta
import json

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingest.data_pipeline import DataPipeline, ingest_all_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run the data ingestion pipeline')
    
    parser.add_argument('--symbols', type=str, nargs='+', default=['AAPL', 'MSFT', 'GOOGL'],
                        help='Stock symbols to ingest data for')
    
    parser.add_argument('--days', type=int, default=30,
                        help='Number of days of historical data to ingest')
    
    parser.add_argument('--interval', type=str, default='1d', choices=['1d', '1h', '15m'],
                        help='Time interval for stock data')
    
    parser.add_argument('--news-keywords', type=str, nargs='+',
                        help='Keywords for news search (defaults to symbols)')
    
    parser.add_argument('--no-delta', action='store_true',
                        help='Do not save data to Delta Lake')
    
    parser.add_argument('--prefer-source', type=str, default='polygon', choices=['polygon', 'yahoo'],
                        help='Preferred data source for stock data')
    
    parser.add_argument('--output', type=str, default='data_pipeline_results.json',
                        help='Output file for ingestion results')
    
    return parser.parse_args()


def main():
    """Run the data ingestion pipeline."""
    args = parse_args()
    
    logger.info(f"Starting data ingestion pipeline for symbols: {args.symbols}")
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)
    
    logger.info(f"Date range: {start_date} to {end_date}")
    
    # Create data pipeline
    pipeline = DataPipeline(
        use_polygon=True,
        use_yahoo=True,
        use_newsapi=True,
        use_cache=True
    )
    
    # Ingest all data
    results = pipeline.ingest_all_data(
        symbols=args.symbols,
        start=start_date,
        end=end_date,
        interval=args.interval,
        news_keywords=args.news_keywords,
        save_to_delta=not args.no_delta
    )
    
    # Save results to file
    with open(args.output, 'w') as f:
        # Convert results to JSON-serializable format
        serializable_results = {
            "stock_data": results["stock_data"],
            "fundamentals": results["fundamentals"],
            "news": results["news"],
            "metadata": {
                "symbols": args.symbols,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "interval": args.interval,
                "news_keywords": args.news_keywords or args.symbols,
                "save_to_delta": not args.no_delta
            }
        }
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Ingestion results saved to {args.output}")
    
    # Print summary
    print("\nData Ingestion Summary:")
    print("======================")
    
    print("\nStock Data:")
    for symbol, data in results["stock_data"].items():
        status = "✓" if data.get("success") else "✗"
        records = data.get("records", 0)
        print(f"  {symbol}: {status} ({records} records)")
    
    print("\nFundamentals:")
    for symbol, data in results["fundamentals"].items():
        status = "✓" if data.get("success") else "✗"
        fields = len(data.get("fields", []))
        print(f"  {symbol}: {status} ({fields} fields)")
    
    print("\nNews Articles:")
    if results["news"] and results["news"].get("success"):
        articles = results["news"].get("articles", 0)
        print(f"  ✓ ({articles} articles)")
    else:
        print("  ✗ (failed)")
    
    print("\nComplete! See data_pipeline.log for details.")


if __name__ == '__main__':
    main()
