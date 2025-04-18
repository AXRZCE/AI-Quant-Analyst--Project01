"""
Batch job for computing sentiment scores from news articles.
"""
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from glob import glob
from typing import List, Dict, Any, Optional, Union
import sys

# Add src directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import local modules
from nlp.finbert_sentiment import FinBERTSentiment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_news_data(source_path: str) -> pd.DataFrame:
    """
    Load news data from Parquet files.
    
    Args:
        source_path: Path to the news data files (can include wildcards)
        
    Returns:
        DataFrame with news data
    """
    logger.info(f"Loading news data from {source_path}")
    
    # Find all files matching the pattern
    files = glob(source_path)
    
    if not files:
        raise ValueError(f"No files found matching {source_path}")
    
    logger.info(f"Found {len(files)} files")
    
    # Load all files
    dfs = []
    for file in files:
        try:
            df = pd.read_parquet(file)
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Error loading {file}: {e}")
    
    if not dfs:
        raise ValueError(f"No data loaded from {source_path}")
    
    # Combine all data
    df = pd.concat(dfs, ignore_index=True)
    
    logger.info(f"Loaded {len(df)} news articles")
    
    return df

def extract_symbols_from_text(text: str, symbols: List[str]) -> List[str]:
    """
    Extract stock symbols from text.
    
    Args:
        text: Text to extract symbols from
        symbols: List of symbols to look for
        
    Returns:
        List of symbols found in the text
    """
    if not text or not isinstance(text, str):
        return []
    
    # Convert text to uppercase for case-insensitive matching
    text_upper = text.upper()
    
    # Find all symbols in the text
    found_symbols = []
    for symbol in symbols:
        if symbol.upper() in text_upper:
            found_symbols.append(symbol)
    
    return found_symbols

def compute_news_sentiment(
    source_path: str,
    output_path: str,
    symbols: Optional[List[str]] = None,
    model_name: str = "yiyanghkust/finbert-tone",
    batch_size: int = 16,
    max_length: int = 512
) -> pd.DataFrame:
    """
    Compute sentiment scores for news articles.
    
    Args:
        source_path: Path to the news data files (can include wildcards)
        output_path: Path to save the sentiment scores
        symbols: List of symbols to extract from the news (if None, use default symbols)
        model_name: Name of the pre-trained model
        batch_size: Batch size for inference
        max_length: Maximum sequence length
        
    Returns:
        DataFrame with sentiment scores
    """
    logger.info("Computing news sentiment")
    
    # Load news data
    df = load_news_data(source_path)
    
    # Set default symbols if not provided
    if symbols is None:
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "WMT"]
    
    # Extract symbols from news articles
    logger.info(f"Extracting symbols from news articles (looking for {len(symbols)} symbols)")
    
    # Combine title and content for better symbol extraction
    if "title" in df.columns and "content" in df.columns:
        df["text"] = df["title"] + " " + df["content"].fillna("")
    elif "title" in df.columns:
        df["text"] = df["title"]
    elif "content" in df.columns:
        df["text"] = df["content"]
    else:
        raise ValueError("News data must have either 'title' or 'content' column")
    
    # Extract symbols
    df["mentioned_symbols"] = df["text"].apply(lambda x: extract_symbols_from_text(x, symbols))
    
    # Filter articles that mention at least one symbol
    df_with_symbols = df[df["mentioned_symbols"].apply(len) > 0]
    
    logger.info(f"Found {len(df_with_symbols)} articles mentioning at least one symbol")
    
    if len(df_with_symbols) == 0:
        logger.warning("No articles found mentioning any symbols")
        return df
    
    # Initialize FinBERT sentiment analyzer
    analyzer = FinBERTSentiment(
        model_name=model_name,
        batch_size=batch_size,
        max_length=max_length
    )
    
    # Analyze sentiment
    logger.info("Analyzing sentiment of news articles")
    
    # Use title for sentiment analysis if available, otherwise use text
    if "title" in df_with_symbols.columns:
        texts = df_with_symbols["title"].tolist()
    else:
        texts = df_with_symbols["text"].tolist()
    
    # Get sentiment scores
    sentiments = analyzer.analyze(texts)
    
    # Add sentiment scores to DataFrame
    sentiment_df = pd.DataFrame(sentiments)
    df_with_symbols = pd.concat([df_with_symbols.reset_index(drop=True), sentiment_df], axis=1)
    
    # Explode the mentioned_symbols column to create one row per symbol
    df_exploded = df_with_symbols.explode("mentioned_symbols").reset_index(drop=True)
    
    # Rename the mentioned_symbols column to symbol
    df_exploded = df_exploded.rename(columns={"mentioned_symbols": "symbol"})
    
    # Add date column if not present
    if "date" not in df_exploded.columns:
        if "published_at" in df_exploded.columns:
            # Try to extract date from published_at
            try:
                df_exploded["date"] = pd.to_datetime(df_exploded["published_at"]).dt.date.astype(str)
            except:
                # If published_at is not a datetime, use current date
                df_exploded["date"] = datetime.utcnow().date().isoformat()
        else:
            # Use current date
            df_exploded["date"] = datetime.utcnow().date().isoformat()
    
    # Add ingest_time column
    df_exploded["ingest_time"] = datetime.utcnow().isoformat()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Save to Parquet
    output_file = f"{output_path}/news_sentiment_{datetime.utcnow().date().isoformat()}.parquet"
    df_exploded.to_parquet(output_file, index=False)
    
    logger.info(f"Saved sentiment scores to {output_file}")
    
    return df_exploded

def aggregate_sentiment_by_symbol(
    df: pd.DataFrame,
    date_column: str = "date"
) -> pd.DataFrame:
    """
    Aggregate sentiment scores by symbol and date.
    
    Args:
        df: DataFrame with sentiment scores
        date_column: Name of the date column
        
    Returns:
        DataFrame with aggregated sentiment scores
    """
    logger.info("Aggregating sentiment scores by symbol and date")
    
    # Group by symbol and date
    agg_df = df.groupby(["symbol", date_column]).agg({
        "positive": "mean",
        "neutral": "mean",
        "negative": "mean",
        "symbol": "count"
    }).rename(columns={"symbol": "mention_count"}).reset_index()
    
    # Calculate sentiment score
    agg_df["sentiment_score"] = agg_df["positive"] - agg_df["negative"]
    
    # Calculate sentiment magnitude
    agg_df["sentiment_magnitude"] = agg_df["positive"] + agg_df["negative"]
    
    # Calculate weighted sentiment score
    agg_df["weighted_sentiment"] = agg_df["sentiment_score"] * agg_df["mention_count"]
    
    return agg_df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute sentiment scores for news articles")
    parser.add_argument("--source-path", type=str, required=True, help="Path to the news data files (can include wildcards)")
    parser.add_argument("--output-path", type=str, default="data/features/batch/sentiment", help="Path to save the sentiment scores")
    parser.add_argument("--symbols", type=str, nargs="+", help="List of symbols to extract from the news")
    parser.add_argument("--model-name", type=str, default="yiyanghkust/finbert-tone", help="Name of the pre-trained model")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length")
    
    args = parser.parse_args()
    
    compute_news_sentiment(
        source_path=args.source_path,
        output_path=args.output_path,
        symbols=args.symbols,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length
    )
