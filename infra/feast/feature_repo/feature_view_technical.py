"""
Feature view definitions for technical indicators.
"""
from datetime import timedelta
from feast import (
    FeatureView, Field, FileSource, 
    ValueType, FeatureService
)
from feast.types import Float32, Float64, Int64

from entities import symbol

# Define the source for batch technical indicators
batch_source = FileSource(
    path="data/features/batch/technical",
    event_timestamp_column="timestamp",
    created_timestamp_column=None,
)

# Define the feature view for technical indicators
technical_indicators_view = FeatureView(
    name="technical_indicators",
    entities=[symbol],
    ttl=timedelta(days=1),
    schema=[
        Field(name="ma_5", dtype=Float32),
        Field(name="ma_15", dtype=Float32),
        Field(name="ma_60", dtype=Float32),
        Field(name="rsi_14", dtype=Float32),
        Field(name="atr_14", dtype=Float32),
    ],
    source=batch_source,
    online=True,
    tags={"category": "technical", "type": "batch"},
    description="Technical indicators computed from historical price data",
)

# Define a feature service for technical indicators
technical_indicators_service = FeatureService(
    name="technical_indicators_service",
    features=[technical_indicators_view],
    description="Service for accessing technical indicators",
)
