"""
Feature view definitions for sentiment features.
"""
from datetime import timedelta
from feast import (
    FeatureView, Field, FileSource, 
    ValueType, FeatureService
)
from feast.types import Float32, Int64

from entities import symbol

# Define the source for sentiment features
sentiment_source = FileSource(
    path="data/features/batch/sentiment",
    event_timestamp_column="published_at",
    created_timestamp_column="ingest_time",
)

# Define the feature view for sentiment features
sentiment_view = FeatureView(
    name="news_sentiment",
    entities=[symbol],
    ttl=timedelta(days=1),
    schema=[
        Field(name="positive", dtype=Float32),
        Field(name="neutral", dtype=Float32),
        Field(name="negative", dtype=Float32),
        Field(name="mention_count", dtype=Int64),
    ],
    source=sentiment_source,
    online=True,
    tags={"category": "sentiment", "type": "batch"},
    description="Sentiment scores from news articles",
)

# Define a feature service for sentiment features
sentiment_service = FeatureService(
    name="sentiment_service",
    features=[sentiment_view],
    description="Service for accessing sentiment features",
)
