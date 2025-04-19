"""
Feast feature registry module for feature management.

This module provides functions for registering and retrieving features from Feast,
ensuring consistent feature access across training and inference.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

import pandas as pd
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_FEAST_REPO_PATH = os.getenv('FEAST_REPO_PATH', 'data/feast')

# Ensure directories exist
os.makedirs(DEFAULT_FEAST_REPO_PATH, exist_ok=True)


class FeastRegistry:
    """Feast feature registry for feature management."""
    
    def __init__(self, repo_path: str = DEFAULT_FEAST_REPO_PATH):
        """
        Initialize the Feast feature registry.
        
        Args:
            repo_path: Path to the Feast feature repository
        """
        logger.info(f"Initializing Feast feature registry at {repo_path}")
        self.repo_path = repo_path
        
        # Create feature repository if it doesn't exist
        self._ensure_feature_repo()
    
    def _ensure_feature_repo(self):
        """Ensure the Feast feature repository exists and is properly configured."""
        try:
            # Check if feature_store.yaml exists
            feature_store_yaml = os.path.join(self.repo_path, "feature_store.yaml")
            if not os.path.exists(feature_store_yaml):
                logger.info("Creating Feast feature repository")
                
                # Create feature_store.yaml
                feature_store_content = """
project: quant_analyst
registry: data/feast/registry.db
provider: local
online_store:
    type: sqlite
    path: data/feast/online_store.db
offline_store:
    type: file
entity_key_serialization_version: 2
"""
                with open(feature_store_yaml, "w") as f:
                    f.write(feature_store_content)
                
                # Create basic entities.py
                entities_path = os.path.join(self.repo_path, "entities.py")
                entities_content = """
from feast import Entity

# Stock entity
stock = Entity(
    name="stock",
    description="Stock ticker symbol",
    join_keys=["symbol"],
)
"""
                with open(entities_path, "w") as f:
                    f.write(entities_content)
                
                # Create basic features directory
                features_dir = os.path.join(self.repo_path, "features")
                os.makedirs(features_dir, exist_ok=True)
                
                # Create __init__.py
                init_path = os.path.join(features_dir, "__init__.py")
                with open(init_path, "w") as f:
                    f.write("")
                
                logger.info("Feast feature repository created successfully")
            else:
                logger.info("Feast feature repository already exists")
        
        except Exception as e:
            logger.error(f"Error ensuring Feast feature repository: {str(e)}")
            raise
    
    def create_feature_view_file(
        self,
        name: str,
        description: str,
        entities: List[str],
        features: List[str],
        source_path: str,
        ttl: int = 86400 * 30  # 30 days in seconds
    ) -> bool:
        """
        Create a feature view file in the Feast repository.
        
        Args:
            name: Name of the feature view
            description: Description of the feature view
            entities: List of entity names
            features: List of feature names
            source_path: Path to the source data
            ttl: Time-to-live in seconds
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create feature view file
            feature_view_path = os.path.join(self.repo_path, "features", f"{name}.py")
            
            # Format feature definitions
            feature_defs = []
            for feature in features:
                feature_defs.append(f'        Field(name="{feature}"),')
            
            feature_defs_str = "\n".join(feature_defs)
            
            # Format entity references
            entity_refs = []
            for entity in entities:
                entity_refs.append(f'        "{entity}",')
            
            entity_refs_str = "\n".join(entity_refs)
            
            # Create feature view content
            feature_view_content = f"""
from datetime import timedelta
from feast import Field, FeatureView, FileSource
from feast.types import Float32, Int64, String
from feast.value_type import ValueType
from feast.on_demand_feature_view import on_demand_feature_view
from feast.request_source import RequestSource
from feast.transformations import window_aggregate
from entities import {", ".join(entities)}

# Source for the {name} feature view
{name}_source = FileSource(
    path="{source_path}",
    timestamp_field="timestamp",
    created_timestamp_column="created",
)

# {name} feature view
{name}_view = FeatureView(
    name="{name}",
    description="{description}",
    entities=[
{entity_refs_str}
    ],
    ttl=timedelta(seconds={ttl}),
    schema=[
{feature_defs_str}
    ],
    source="{name}_source",
    online=True,
    tags={{"team": "quant_analyst"}},
)
"""
            
            # Write feature view file
            with open(feature_view_path, "w") as f:
                f.write(feature_view_content)
            
            logger.info(f"Created feature view file: {feature_view_path}")
            
            # Update __init__.py to import the new feature view
            init_path = os.path.join(self.repo_path, "features", "__init__.py")
            with open(init_path, "a") as f:
                f.write(f"from .{name} import {name}_view\n")
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating feature view file: {str(e)}")
            return False
    
    def register_features(self) -> bool:
        """
        Register features with Feast.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Change to the repository directory
            original_dir = os.getcwd()
            os.chdir(self.repo_path)
            
            # Run feast apply
            import subprocess
            result = subprocess.run(["feast", "apply"], capture_output=True, text=True)
            
            # Change back to the original directory
            os.chdir(original_dir)
            
            if result.returncode == 0:
                logger.info("Features registered successfully")
                return True
            else:
                logger.error(f"Error registering features: {result.stderr}")
                return False
            
        except Exception as e:
            logger.error(f"Error registering features: {str(e)}")
            return False
    
    def create_price_features(self, source_path: str) -> bool:
        """
        Create price features in the Feast repository.
        
        Args:
            source_path: Path to the source data
            
        Returns:
            True if successful, False otherwise
        """
        return self.create_feature_view_file(
            name="price",
            description="Stock price features",
            entities=["stock"],
            features=["open", "high", "low", "close", "volume", "vwap"],
            source_path=source_path
        )
    
    def create_technical_features(self, source_path: str) -> bool:
        """
        Create technical indicator features in the Feast repository.
        
        Args:
            source_path: Path to the source data
            
        Returns:
            True if successful, False otherwise
        """
        return self.create_feature_view_file(
            name="technical",
            description="Technical indicator features",
            entities=["stock"],
            features=[
                "sma_5", "sma_10", "sma_20", "sma_50", "sma_200",
                "ema_5", "ema_10", "ema_20", "ema_50", "ema_200",
                "rsi_14", "macd", "macd_signal", "macd_hist",
                "bollinger_upper", "bollinger_middle", "bollinger_lower",
                "atr_14", "adx_14"
            ],
            source_path=source_path
        )
    
    def create_fundamental_features(self, source_path: str) -> bool:
        """
        Create fundamental features in the Feast repository.
        
        Args:
            source_path: Path to the source data
            
        Returns:
            True if successful, False otherwise
        """
        return self.create_feature_view_file(
            name="fundamental",
            description="Fundamental features",
            entities=["stock"],
            features=[
                "revenue", "net_income", "eps", "pe_ratio",
                "price_to_book", "dividend_yield", "market_cap",
                "debt_to_equity", "return_on_equity", "return_on_assets"
            ],
            source_path=source_path
        )
    
    def create_sentiment_features(self, source_path: str) -> bool:
        """
        Create sentiment features in the Feast repository.
        
        Args:
            source_path: Path to the source data
            
        Returns:
            True if successful, False otherwise
        """
        return self.create_feature_view_file(
            name="sentiment",
            description="News sentiment features",
            entities=["stock"],
            features=[
                "sentiment_score", "sentiment_magnitude",
                "positive_news_count", "negative_news_count",
                "neutral_news_count", "news_volume"
            ],
            source_path=source_path
        )


# Singleton instance for easy access
_feast_registry = None

def get_feast_registry() -> FeastRegistry:
    """
    Get the singleton instance of FeastRegistry.
    
    Returns:
        FeastRegistry instance
    """
    global _feast_registry
    if _feast_registry is None:
        _feast_registry = FeastRegistry()
    return _feast_registry


def create_price_features(source_path: str) -> bool:
    """
    Convenience function to create price features in the Feast repository.
    
    Args:
        source_path: Path to the source data
        
    Returns:
        True if successful, False otherwise
    """
    return get_feast_registry().create_price_features(source_path)


def create_technical_features(source_path: str) -> bool:
    """
    Convenience function to create technical indicator features in the Feast repository.
    
    Args:
        source_path: Path to the source data
        
    Returns:
        True if successful, False otherwise
    """
    return get_feast_registry().create_technical_features(source_path)


def create_fundamental_features(source_path: str) -> bool:
    """
    Convenience function to create fundamental features in the Feast repository.
    
    Args:
        source_path: Path to the source data
        
    Returns:
        True if successful, False otherwise
    """
    return get_feast_registry().create_fundamental_features(source_path)


def create_sentiment_features(source_path: str) -> bool:
    """
    Convenience function to create sentiment features in the Feast repository.
    
    Args:
        source_path: Path to the source data
        
    Returns:
        True if successful, False otherwise
    """
    return get_feast_registry().create_sentiment_features(source_path)


def register_features() -> bool:
    """
    Convenience function to register features with Feast.
    
    Returns:
        True if successful, False otherwise
    """
    return get_feast_registry().register_features()
