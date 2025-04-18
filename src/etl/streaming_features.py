"""
Streaming ETL job for computing real-time features from tick data.
"""
import os
import logging
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
import pyspark.sql.functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, 
    TimestampType, IntegerType
)
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
KAFKA_BOOTSTRAP = os.getenv('KAFKA_BOOTSTRAP', 'localhost:9092')
FEATURES_DIR = os.getenv('FEATURES_DIR', 'data/features')

def init_spark():
    """Initialize Spark session."""
    return (
        SparkSession.builder
        .appName("streaming_features")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.sql.streaming.checkpointLocation", f"{FEATURES_DIR}/streaming/checkpoints")
        .getOrCreate()
    )

def define_schema():
    """Define schema for incoming tick data."""
    return StructType([
        StructField("timestamp", TimestampType(), True),
        StructField("open", DoubleType(), True),
        StructField("high", DoubleType(), True),
        StructField("low", DoubleType(), True),
        StructField("close", DoubleType(), True),
        StructField("volume", IntegerType(), True),
        StructField("symbol", StringType(), True)
    ])

def read_from_kafka(spark):
    """Read streaming data from Kafka.
    
    Args:
        spark: Spark session
        
    Returns:
        DataFrame: Streaming DataFrame
    """
    logger.info(f"Reading from Kafka topic 'ticks' at {KAFKA_BOOTSTRAP}")
    
    schema = define_schema()
    
    df = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
        .option("subscribe", "ticks")
        .option("startingOffsets", "latest")
        .load()
    )
    
    # Parse JSON value
    parsed_df = df.select(
        F.from_json(F.col("value").cast("string"), schema).alias("data")
    ).select("data.*")
    
    # Ensure timestamp is in the correct format
    parsed_df = parsed_df.withColumn(
        "timestamp", 
        F.when(F.col("timestamp").isNotNull(), F.col("timestamp"))
         .otherwise(F.current_timestamp())
    )
    
    # Extract date for partitioning
    parsed_df = parsed_df.withColumn("date", F.to_date("timestamp"))
    
    return parsed_df

def read_from_delta(spark, test_mode=False):
    """Read streaming data from Delta Lake.
    
    Args:
        spark: Spark session
        test_mode: Whether to run in test mode
        
    Returns:
        DataFrame: Streaming DataFrame
    """
    if test_mode:
        source_path = "data/raw/ticks"
    else:
        source_path = "data/raw/ticks"
    
    logger.info(f"Reading from Delta Lake at {source_path}")
    
    df = (
        spark.readStream
        .format("delta")
        .option("ignoreDeletes", "true")
        .option("ignoreChanges", "true")
        .load(source_path)
    )
    
    # Ensure timestamp is in the correct format
    df = df.withColumn(
        "timestamp", 
        F.when(F.col("timestamp").isNotNull(), F.col("timestamp"))
         .otherwise(F.current_timestamp())
    )
    
    # Extract date for partitioning
    df = df.withColumn("date", F.to_date("timestamp"))
    
    return df

def compute_streaming_features(df):
    """Compute streaming features from tick data.
    
    Args:
        df: Streaming DataFrame with tick data
        
    Returns:
        DataFrame: DataFrame with streaming features
    """
    logger.info("Computing streaming features")
    
    # Define windows for different time periods
    # Note: For streaming, we use time-based windows rather than row-based
    w1min = (
        Window.partitionBy("symbol")
        .orderBy("timestamp")
        .rangeBetween(-60, 0)  # 1 minute in seconds
    )
    
    w5min = (
        Window.partitionBy("symbol")
        .orderBy("timestamp")
        .rangeBetween(-300, 0)  # 5 minutes in seconds
    )
    
    # Compute VWAP (Volume-Weighted Average Price)
    df = df.withColumn(
        "price_volume", 
        F.col("close") * F.col("volume")
    )
    
    df = df.withColumn(
        "vwap_1m",
        F.sum("price_volume").over(w1min) / F.sum("volume").over(w1min)
    )
    
    df = df.withColumn(
        "vwap_5m",
        F.sum("price_volume").over(w5min) / F.sum("volume").over(w5min)
    )
    
    # Compute rolling volatility (standard deviation of returns)
    df = df.withColumn(
        "return", 
        F.log(F.col("close") / F.lag("close", 1).over(Window.partitionBy("symbol").orderBy("timestamp")))
    )
    
    df = df.withColumn(
        "volatility_1m",
        F.stddev("return").over(w1min)
    )
    
    df = df.withColumn(
        "volatility_5m",
        F.stddev("return").over(w5min)
    )
    
    # Compute price momentum
    df = df.withColumn(
        "momentum_1m",
        F.col("close") / F.first("close").over(w1min) - 1
    )
    
    df = df.withColumn(
        "momentum_5m",
        F.col("close") / F.first("close").over(w5min) - 1
    )
    
    # Select relevant columns
    result_df = df.select(
        "symbol",
        "timestamp",
        "date",
        "close",
        "volume",
        "vwap_1m",
        "vwap_5m",
        "volatility_1m",
        "volatility_5m",
        "momentum_1m",
        "momentum_5m"
    )
    
    return result_df

def write_to_delta(df, output_dir=None, test_mode=False):
    """Write streaming features to Delta Lake.
    
    Args:
        df: DataFrame with streaming features
        output_dir: Output directory
        test_mode: Whether to run in test mode
    """
    if output_dir is None:
        if test_mode:
            output_dir = f"{FEATURES_DIR}/streaming_test"
        else:
            output_dir = f"{FEATURES_DIR}/streaming/rolling"
    
    logger.info(f"Writing streaming features to {output_dir}")
    
    checkpoint_dir = f"{output_dir}/checkpoints"
    
    # Write the streaming DataFrame to Delta Lake
    query = (
        df.writeStream
        .format("delta")
        .outputMode("append")
        .option("checkpointLocation", checkpoint_dir)
        .partitionBy("date")
        .trigger(processingTime="10 seconds")  # Process every 10 seconds
        .start(output_dir)
    )
    
    return query

def write_to_kafka(df):
    """Write streaming features to Kafka.
    
    Args:
        df: DataFrame with streaming features
    """
    logger.info(f"Writing streaming features to Kafka topic 'rolling_features'")
    
    # Convert DataFrame to JSON string
    df_json = df.select(
        F.to_json(F.struct("*")).alias("value")
    )
    
    # Write to Kafka
    query = (
        df_json.writeStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
        .option("topic", "rolling_features")
        .option("checkpointLocation", f"{FEATURES_DIR}/streaming/kafka_checkpoints")
        .start()
    )
    
    return query

def main(source="kafka", sink="delta", test_mode=False):
    """Main ETL function.
    
    Args:
        source: Source type ('kafka' or 'delta')
        sink: Sink type ('delta' or 'kafka')
        test_mode: Whether to run in test mode
    """
    logger.info(f"Starting streaming feature computation (source={source}, sink={sink}, test_mode={test_mode})")
    
    # Initialize Spark
    spark = init_spark()
    
    try:
        # Read from source
        if source == "kafka":
            df = read_from_kafka(spark)
        else:
            df = read_from_delta(spark, test_mode)
        
        # Compute streaming features
        result_df = compute_streaming_features(df)
        
        # Write to sink
        if sink == "delta":
            query = write_to_delta(result_df, test_mode=test_mode)
        else:
            query = write_to_kafka(result_df)
        
        logger.info("Streaming query started, waiting for termination")
        
        # Wait for the query to terminate
        query.awaitTermination()
    except Exception as e:
        logger.error(f"Error in streaming feature computation: {e}")
        raise
    finally:
        # Stop Spark session
        spark.stop()

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Streaming ETL job for computing real-time features')
    parser.add_argument('--source', choices=['kafka', 'delta'], default='kafka', help='Source type')
    parser.add_argument('--sink', choices=['delta', 'kafka'], default='delta', help='Sink type')
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    
    args = parser.parse_args()
    
    main(args.source, args.sink, args.test)
