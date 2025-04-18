"""
Feature view definitions for fundamental indicators.
"""
from datetime import timedelta
from feast import (
    FeatureView, Field, FileSource, 
    ValueType, FeatureService
)
from feast.types import Float32, Float64, Int64, String

from entities import symbol

# Define the source for batch fundamental indicators
batch_source = FileSource(
    path="data/features/batch/fundamental",
    event_timestamp_column="date",
    created_timestamp_column=None,
)

# Define the feature view for fundamental indicators
fundamental_indicators_view = FeatureView(
    name="fundamental_indicators",
    entities=[symbol],
    ttl=timedelta(days=30),  # Fundamental data changes less frequently
    schema=[
        Field(name="price_to_earnings", dtype=Float32),
        Field(name="debt_to_equity", dtype=Float32),
        Field(name="earnings_surprise", dtype=Float32),
        Field(name="pe_category", dtype=String),
        Field(name="avg_sentiment", dtype=Float32),
        Field(name="mention_count", dtype=Int64),
    ],
    source=batch_source,
    online=True,
    tags={"category": "fundamental", "type": "batch"},
    description="Fundamental indicators computed from financial data and news",
)

# Define a feature service for fundamental indicators
fundamental_indicators_service = FeatureService(
    name="fundamental_indicators_service",
    features=[fundamental_indicators_view],
    description="Service for accessing fundamental indicators",
)
