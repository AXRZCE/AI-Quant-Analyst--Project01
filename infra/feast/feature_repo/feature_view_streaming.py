"""
Feature view definitions for streaming features.
"""
from datetime import timedelta
from feast import (
    FeatureView, Field, FileSource, KafkaSource,
    ValueType, FeatureService
)
from feast.types import Float32, Float64, Int64
from feast.data_format import JsonFormat

from entities import symbol

# Define the source for streaming features
# For local testing, we'll use a file source
file_source = FileSource(
    path="data/features/streaming/rolling",
    event_timestamp_column="timestamp",
    created_timestamp_column=None,
)

# Define Kafka source for production
kafka_source = KafkaSource(
    name="kafka_streaming_source",
    kafka_bootstrap_servers="localhost:9092",
    topic="rolling_features",
    event_timestamp_column="timestamp",
    message_format=JsonFormat(
        schema_json="""
        {
            "type": "record",
            "name": "rolling_features",
            "fields": [
                {"name": "symbol", "type": "string"},
                {"name": "timestamp", "type": "string"},
                {"name": "date", "type": "string"},
                {"name": "close", "type": "double"},
                {"name": "volume", "type": "int"},
                {"name": "vwap_1m", "type": "double"},
                {"name": "vwap_5m", "type": "double"},
                {"name": "volatility_1m", "type": "double"},
                {"name": "volatility_5m", "type": "double"},
                {"name": "momentum_1m", "type": "double"},
                {"name": "momentum_5m", "type": "double"}
            ]
        }
        """
    ),
)

# Define the feature view for streaming features
streaming_features_view = FeatureView(
    name="streaming_features",
    entities=[symbol],
    ttl=timedelta(minutes=10),  # Streaming data is short-lived
    schema=[
        Field(name="vwap_1m", dtype=Float32),
        Field(name="vwap_5m", dtype=Float32),
        Field(name="volatility_1m", dtype=Float32),
        Field(name="volatility_5m", dtype=Float32),
        Field(name="momentum_1m", dtype=Float32),
        Field(name="momentum_5m", dtype=Float32),
    ],
    source=file_source,  # Use file_source for local testing, kafka_source for production
    online=True,
    tags={"category": "streaming", "type": "real-time"},
    description="Real-time streaming features computed from tick data",
)

# Define a feature service for streaming features
streaming_features_service = FeatureService(
    name="streaming_features_service",
    features=[streaming_features_view],
    description="Service for accessing real-time streaming features",
)
