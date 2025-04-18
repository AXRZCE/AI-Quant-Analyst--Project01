"""
Batch ETL job for computing fundamental indicators from news and financial data.
"""
import os
import logging
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType, StringType, DateType
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
RAW_DATA_DIR = os.getenv('RAW_DATA_DIR', 'data/raw')
FEATURES_DIR = os.getenv('FEATURES_DIR', 'data/features')

def init_spark():
    """Initialize Spark session."""
    return (
        SparkSession.builder
        .appName("batch_fundamentals")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .getOrCreate()
    )

def load_news_data(spark, date=None):
    """Load news data from Parquet files.
    
    Args:
        spark: Spark session
        date: Optional date filter (YYYY-MM-DD)
        
    Returns:
        DataFrame: Loaded news data
    """
    if date:
        path = f"{RAW_DATA_DIR}/news/{date}/*.parquet"
    else:
        path = f"{RAW_DATA_DIR}/news/*/*.parquet"
    
    logger.info(f"Loading news data from {path}")
    
    try:
        df = spark.read.parquet(path)
        
        # Convert timestamp to datetime if it's a string
        if df.schema["timestamp"].dataType.typeName() == "string":
            df = df.withColumn("timestamp", F.to_timestamp("timestamp"))
        
        # Extract date from timestamp for partitioning
        df = df.withColumn("date", F.to_date("timestamp"))
        
        return df
    except Exception as e:
        logger.warning(f"Failed to load news data: {e}")
        return None

def load_fundamental_data(spark):
    """Load fundamental financial data.
    
    In a real-world scenario, this would load from a financial data provider.
    For this example, we'll create synthetic data.
    
    Args:
        spark: Spark session
        
    Returns:
        DataFrame: Fundamental financial data
    """
    logger.info("Creating synthetic fundamental data")
    
    # Create a synthetic dataset with fundamental metrics
    data = [
        # symbol, date, pe_ratio, debt_equity_ratio, earnings_surprise
        ("AAPL", "2025-04-17", 25.6, 1.2, 0.05),
        ("MSFT", "2025-04-17", 30.2, 0.8, 0.03),
        ("GOOGL", "2025-04-17", 28.4, 0.5, 0.02),
        ("AMZN", "2025-04-17", 35.1, 1.5, -0.01),
        ("META", "2025-04-17", 22.3, 0.7, 0.04)
    ]
    
    schema = ["symbol", "date", "pe_ratio", "debt_equity_ratio", "earnings_surprise"]
    df = spark.createDataFrame(data, schema=schema)
    
    # Convert date string to date type
    df = df.withColumn("date", F.to_date("date"))
    
    return df

def compute_sentiment_scores(news_df):
    """Compute sentiment scores from news data.
    
    In a real-world scenario, this would use NLP models for sentiment analysis.
    For this example, we'll create synthetic sentiment scores.
    
    Args:
        news_df: DataFrame with news data
        
    Returns:
        DataFrame: News data with sentiment scores
    """
    if news_df is None:
        return None
    
    logger.info("Computing sentiment scores from news data")
    
    # Add a simple random sentiment score between -1 and 1
    # In a real implementation, this would use NLP models
    news_df = news_df.withColumn(
        "sentiment_score", 
        (F.rand() * 2) - 1  # Random value between -1 and 1
    )
    
    # Extract company symbols from news content
    # This is a simplified approach - in reality, you'd use NER models
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    
    # Create a UDF to extract symbols from text
    def extract_symbols_from_text(text, symbols_list):
        if not text:
            return []
        return [symbol for symbol in symbols_list if symbol in text]
    
    extract_symbols_udf = F.udf(
        lambda text: extract_symbols_from_text(text, symbols),
        returnType=F.ArrayType(StringType())
    )
    
    # Apply the UDF to extract symbols from title and content
    news_df = news_df.withColumn(
        "mentioned_symbols",
        extract_symbols_udf(F.concat_ws(" ", "title", "content"))
    )
    
    # Explode the array to create one row per symbol mention
    news_df = news_df.withColumn("symbol", F.explode("mentioned_symbols"))
    
    # Aggregate sentiment by symbol and date
    sentiment_df = news_df.groupBy("symbol", "date").agg(
        F.avg("sentiment_score").alias("avg_sentiment"),
        F.count("*").alias("mention_count")
    )
    
    return sentiment_df

def compute_fundamental_indicators(fundamental_df, sentiment_df=None):
    """Compute fundamental indicators by combining financial data and news sentiment.
    
    Args:
        fundamental_df: DataFrame with fundamental financial data
        sentiment_df: Optional DataFrame with news sentiment data
        
    Returns:
        DataFrame: DataFrame with fundamental indicators
    """
    logger.info("Computing fundamental indicators")
    
    result_df = fundamental_df
    
    # Add derived metrics
    result_df = result_df.withColumn(
        "price_to_earnings", 
        result_df["pe_ratio"]
    )
    
    result_df = result_df.withColumn(
        "debt_to_equity", 
        result_df["debt_equity_ratio"]
    )
    
    # Categorize P/E ratio
    result_df = result_df.withColumn(
        "pe_category",
        F.when(F.col("pe_ratio") < 15, "low")
         .when(F.col("pe_ratio") < 25, "medium")
         .otherwise("high")
    )
    
    # Join with sentiment data if available
    if sentiment_df is not None:
        result_df = result_df.join(
            sentiment_df,
            on=["symbol", "date"],
            how="left"
        )
        
        # Fill missing sentiment values
        result_df = result_df.na.fill(0, subset=["avg_sentiment", "mention_count"])
    else:
        # Add placeholder columns
        result_df = result_df.withColumn("avg_sentiment", F.lit(0.0))
        result_df = result_df.withColumn("mention_count", F.lit(0))
    
    # Select relevant columns
    result_df = result_df.select(
        "symbol",
        "date",
        "price_to_earnings",
        "debt_to_equity",
        "earnings_surprise",
        "pe_category",
        "avg_sentiment",
        "mention_count"
    )
    
    return result_df

def save_fundamental_indicators(df, output_dir=None, mode="overwrite"):
    """Save fundamental indicators to Delta/Parquet format.
    
    Args:
        df: DataFrame with fundamental indicators
        output_dir: Output directory
        mode: Write mode (overwrite, append)
    """
    if output_dir is None:
        output_dir = f"{FEATURES_DIR}/batch/fundamental"
    
    logger.info(f"Saving fundamental indicators to {output_dir}")
    
    try:
        # Try to save as Delta format
        df.write.format("delta").mode(mode).partitionBy("date").save(output_dir)
        logger.info("Saved data in Delta format")
    except Exception as e:
        logger.warning(f"Failed to save in Delta format: {e}")
        logger.info("Falling back to Parquet format")
        
        # Fallback to Parquet format
        df.write.format("parquet").mode(mode).partitionBy("date").save(output_dir)
        logger.info("Saved data in Parquet format")

def main(date=None):
    """Main ETL function.
    
    Args:
        date: Optional date filter (YYYY-MM-DD)
    """
    logger.info("Starting batch fundamental computation")
    
    # Initialize Spark
    spark = init_spark()
    
    try:
        # Load news data
        news_df = load_news_data(spark, date)
        
        # Compute sentiment scores
        sentiment_df = compute_sentiment_scores(news_df)
        
        # Load fundamental data
        fundamental_df = load_fundamental_data(spark)
        
        # Compute fundamental indicators
        result_df = compute_fundamental_indicators(fundamental_df, sentiment_df)
        
        # Save results
        save_fundamental_indicators(result_df)
        
        logger.info("Batch fundamental computation completed successfully")
    except Exception as e:
        logger.error(f"Error in batch fundamental computation: {e}")
        raise
    finally:
        # Stop Spark session
        spark.stop()

if __name__ == "__main__":
    import sys
    
    # Get date from command line argument if provided
    date = sys.argv[1] if len(sys.argv) > 1 else None
    
    main(date)
