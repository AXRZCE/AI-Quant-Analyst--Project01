"""
Batch ETL job for computing technical indicators from raw tick data.
"""
import os
import logging
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType, DateType
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
        .appName("batch_features")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .getOrCreate()
    )

def load_tick_data(spark, date=None):
    """Load tick data from Parquet files.
    
    Args:
        spark: Spark session
        date: Optional date filter (YYYY-MM-DD)
        
    Returns:
        DataFrame: Loaded tick data
    """
    if date:
        path = f"{RAW_DATA_DIR}/ticks/{date}/*.parquet"
    else:
        path = f"{RAW_DATA_DIR}/ticks/*/*.parquet"
    
    logger.info(f"Loading tick data from {path}")
    
    df = spark.read.parquet(path)
    
    # Convert timestamp to datetime if it's a string
    if df.schema["timestamp"].dataType.typeName() == "string":
        df = df.withColumn("timestamp", F.to_timestamp("timestamp"))
    
    # Extract date from timestamp for partitioning
    df = df.withColumn("date", F.to_date("timestamp"))
    
    return df

def compute_technical_indicators(df):
    """Compute technical indicators from tick data.
    
    Args:
        df: DataFrame with tick data
        
    Returns:
        DataFrame: DataFrame with technical indicators
    """
    logger.info("Computing technical indicators")
    
    # Define windows for different periods
    w5 = Window.partitionBy("symbol").orderBy("timestamp").rowsBetween(-4, 0)
    w15 = Window.partitionBy("symbol").orderBy("timestamp").rowsBetween(-14, 0)
    w60 = Window.partitionBy("symbol").orderBy("timestamp").rowsBetween(-59, 0)
    
    # Moving Averages
    df = df.withColumn("ma_5", F.avg("close").over(w5))
    df = df.withColumn("ma_15", F.avg("close").over(w15))
    df = df.withColumn("ma_60", F.avg("close").over(w60))
    
    # RSI - 14 period
    # Step 1: Calculate price changes
    df = df.withColumn("price_change", F.col("close") - F.lag("close", 1).over(Window.partitionBy("symbol").orderBy("timestamp")))
    
    # Step 2: Calculate gains (positive changes) and losses (negative changes)
    df = df.withColumn("gain", F.when(F.col("price_change") > 0, F.col("price_change")).otherwise(0))
    df = df.withColumn("loss", F.when(F.col("price_change") < 0, -F.col("price_change")).otherwise(0))
    
    # Step 3: Calculate average gains and losses over 14 periods
    w_rsi = Window.partitionBy("symbol").orderBy("timestamp").rowsBetween(-13, 0)
    df = df.withColumn("avg_gain", F.avg("gain").over(w_rsi))
    df = df.withColumn("avg_loss", F.avg("loss").over(w_rsi))
    
    # Step 4: Calculate RS (Relative Strength) and RSI
    df = df.withColumn("rs", 
                      F.when(F.col("avg_loss") == 0, 100)
                       .otherwise(F.col("avg_gain") / F.col("avg_loss")))
    
    df = df.withColumn("rsi_14", 
                      F.when(F.col("avg_loss") == 0, 100)
                       .otherwise(100 - (100 / (1 + F.col("rs")))))
    
    # ATR - 14 period
    # Step 1: Calculate True Range
    df = df.withColumn("high_low", F.col("high") - F.col("low"))
    df = df.withColumn("high_close_prev", 
                      F.abs(F.col("high") - F.lag("close", 1).over(Window.partitionBy("symbol").orderBy("timestamp"))))
    df = df.withColumn("low_close_prev", 
                      F.abs(F.col("low") - F.lag("close", 1).over(Window.partitionBy("symbol").orderBy("timestamp"))))
    
    df = df.withColumn("tr", 
                      F.greatest("high_low", "high_close_prev", "low_close_prev"))
    
    # Step 2: Calculate ATR (14-period average of True Range)
    df = df.withColumn("atr_14", F.avg("tr").over(w_rsi))
    
    # Select relevant columns
    result_df = df.select(
        "symbol", 
        "timestamp", 
        "date",
        "open", 
        "high", 
        "low", 
        "close", 
        "volume",
        "ma_5", 
        "ma_15", 
        "ma_60",
        "rsi_14",
        "atr_14"
    )
    
    return result_df

def save_technical_indicators(df, output_dir=None, mode="overwrite"):
    """Save technical indicators to Delta/Parquet format.
    
    Args:
        df: DataFrame with technical indicators
        output_dir: Output directory
        mode: Write mode (overwrite, append)
    """
    if output_dir is None:
        output_dir = f"{FEATURES_DIR}/batch/technical"
    
    logger.info(f"Saving technical indicators to {output_dir}")
    
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
    logger.info("Starting batch feature computation")
    
    # Initialize Spark
    spark = init_spark()
    
    try:
        # Load tick data
        df = load_tick_data(spark, date)
        
        # Compute technical indicators
        result_df = compute_technical_indicators(df)
        
        # Save results
        save_technical_indicators(result_df)
        
        logger.info("Batch feature computation completed successfully")
    except Exception as e:
        logger.error(f"Error in batch feature computation: {e}")
        raise
    finally:
        # Stop Spark session
        spark.stop()

if __name__ == "__main__":
    import sys
    
    # Get date from command line argument if provided
    date = sys.argv[1] if len(sys.argv) > 1 else None
    
    main(date)
