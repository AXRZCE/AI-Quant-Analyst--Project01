"""
Script to consume messages from Kafka topics and save them as Parquet files.
"""
import os
import json
import logging
import time
from datetime import datetime
import pandas as pd
from kafka import KafkaConsumer
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
RAW_DATA_DIR = os.getenv('RAW_DATA_DIR', 'data/raw')
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 500))

def ensure_directory(directory):
    """Ensure a directory exists.

    Args:
        directory (str): Directory path.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

def save_to_parquet(data, topic, timestamp=None):
    """Save data to a Parquet file with date-based partitioning.

    Args:
        data (list): List of records.
        topic (str): Kafka topic name.
        timestamp (str, optional): Timestamp for filename. Defaults to current time.

    Returns:
        str: Path to the saved file.
    """
    if not data:
        logger.warning(f"No data to save for topic {topic}")
        return None

    # Get current date for partitioning
    current_date = datetime.now().strftime('%Y-%m-%d')

    # Create directory structure with date partitioning
    topic_dir = os.path.join(RAW_DATA_DIR, topic, current_date)
    ensure_directory(topic_dir)

    # Generate filename with timestamp
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    filename = f"{topic}_{timestamp}.parquet"
    filepath = os.path.join(topic_dir, filename)

    # Convert to DataFrame and save
    df = pd.DataFrame(data)
    df.to_parquet(filepath, index=False)
    logger.info(f"Saved {len(data)} records to {filepath}")

    return filepath

def consume_and_save(topics=None, timeout_ms=30000, max_records=None):
    """Consume messages from Kafka topics and save as Parquet files.

    Args:
        topics (list, optional): List of topics to consume from. Defaults to ['ticks', 'news', 'tweets'].
        timeout_ms (int, optional): Consumer timeout in milliseconds. Defaults to 30000.
        max_records (int, optional): Maximum number of records to consume per topic. Defaults to BATCH_SIZE.

    Returns:
        dict: Summary of records saved per topic.
    """
    if topics is None:
        topics = ['ticks', 'news', 'tweets']

    if max_records is None:
        max_records = BATCH_SIZE

    # Initialize consumer
    consumer = KafkaConsumer(
        *topics,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        group_id='parquet_converter',
        consumer_timeout_ms=timeout_ms
    )

    # Collect messages by topic
    messages = {topic: [] for topic in topics}
    record_counts = {topic: 0 for topic in topics}

    # Start consuming
    logger.info(f"Starting to consume from topics: {topics}")
    start_time = time.time()

    try:
        for message in consumer:
            topic = message.topic
            messages[topic].append(message.value)
            record_counts[topic] += 1

            # Save batch if we've reached max_records
            if len(messages[topic]) >= max_records:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                save_to_parquet(messages[topic], topic, timestamp)
                messages[topic] = []

            # Log progress
            if sum(record_counts.values()) % 1000 == 0:
                logger.info(f"Consumed {sum(record_counts.values())} records so far")

    except KeyboardInterrupt:
        logger.info("Consumption stopped by user")
    except Exception as e:
        logger.error(f"Error consuming messages: {e}")
    finally:
        # Save any remaining messages
        for topic, data in messages.items():
            if data:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                save_to_parquet(data, topic, timestamp)

        # Close consumer
        consumer.close()

    # Log summary
    duration = time.time() - start_time
    logger.info(f"Consumption completed in {duration:.2f} seconds")
    for topic, count in record_counts.items():
        logger.info(f"Topic {topic}: {count} records")

    return record_counts

def main():
    """Main function to demonstrate usage."""
    # Ensure data directories exist
    for topic in ['ticks', 'news', 'tweets']:
        ensure_directory(os.path.join(RAW_DATA_DIR, topic))

    # Consume and save
    record_counts = consume_and_save(timeout_ms=60000)  # 1 minute timeout

    # Print summary
    logger.info("Summary of saved records:")
    for topic, count in record_counts.items():
        logger.info(f"{topic}: {count} records")

if __name__ == "__main__":
    main()
