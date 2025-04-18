"""
Feature service definitions for combining multiple feature views.
"""
from feast import FeatureService

from feature_view_technical import technical_indicators_view
from feature_view_fundamental import fundamental_indicators_view
from feature_view_streaming import streaming_features_view
from feature_sentiment import sentiment_view

# Define a combined feature service for all features
combined_features_service = FeatureService(
    name="combined_features",
    features=[
        technical_indicators_view,
        fundamental_indicators_view,
        streaming_features_view,
        sentiment_view
    ],
    description="Combined service for accessing all features",
)
