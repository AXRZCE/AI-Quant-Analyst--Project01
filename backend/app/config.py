"""
Configuration module for the API.

This module loads configuration from environment variables and provides
configuration objects for different components of the API.
"""

import os
import secrets
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseSettings, Field, validator
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class SecuritySettings(BaseSettings):
    """Security settings for the API."""
    
    # API key settings
    API_KEY_HEADER: str = Field("X-API-Key", description="Header name for API key authentication")
    API_KEYS: List[str] = Field(
        default_factory=lambda: os.environ.get("API_KEYS", "").split(",") if os.environ.get("API_KEYS") else []
    )
    
    # JWT settings
    JWT_SECRET_KEY: str = Field(
        default_factory=lambda: os.environ.get("JWT_SECRET_KEY", secrets.token_urlsafe(32))
    )
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default_factory=lambda: int(os.environ.get("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", 30))
    )
    
    # CORS settings
    CORS_ORIGINS: List[str] = Field(
        default_factory=lambda: os.environ.get("CORS_ORIGINS", "*").split(",")
    )
    CORS_ALLOW_CREDENTIALS: bool = Field(
        default_factory=lambda: os.environ.get("CORS_ALLOW_CREDENTIALS", "True").lower() == "true"
    )
    CORS_ALLOW_METHODS: List[str] = Field(
        default_factory=lambda: os.environ.get("CORS_ALLOW_METHODS", "*").split(",")
    )
    CORS_ALLOW_HEADERS: List[str] = Field(
        default_factory=lambda: os.environ.get("CORS_ALLOW_HEADERS", "*").split(",")
    )
    
    @validator("API_KEYS")
    def validate_api_keys(cls, v):
        """Validate API keys."""
        return [key.strip() for key in v if key.strip()]
    
    class Config:
        env_file = ".env"
        env_prefix = "SECURITY_"


class RateLimitSettings(BaseSettings):
    """Rate limit settings for the API."""
    
    ENABLED: bool = Field(
        default_factory=lambda: os.environ.get("RATE_LIMIT_ENABLED", "True").lower() == "true"
    )
    RATE_LIMIT_PER_MINUTE: int = Field(
        default_factory=lambda: int(os.environ.get("RATE_LIMIT_PER_MINUTE", 60))
    )
    RATE_LIMIT_STRATEGY: str = Field(
        default_factory=lambda: os.environ.get("RATE_LIMIT_STRATEGY", "fixed-window")
    )
    RATE_LIMIT_REDIS_URL: Optional[str] = Field(
        default_factory=lambda: os.environ.get("RATE_LIMIT_REDIS_URL")
    )
    
    class Config:
        env_file = ".env"
        env_prefix = "RATE_LIMIT_"


class CacheSettings(BaseSettings):
    """Cache settings for the API."""
    
    ENABLED: bool = Field(
        default_factory=lambda: os.environ.get("CACHE_ENABLED", "True").lower() == "true"
    )
    TTL_SECONDS: int = Field(
        default_factory=lambda: int(os.environ.get("CACHE_TTL_SECONDS", 300))
    )
    REDIS_URL: Optional[str] = Field(
        default_factory=lambda: os.environ.get("CACHE_REDIS_URL")
    )
    BACKEND: str = Field(
        default_factory=lambda: os.environ.get("CACHE_BACKEND", "memory")
    )
    MAX_SIZE: int = Field(
        default_factory=lambda: int(os.environ.get("CACHE_MAX_SIZE", 1000))
    )
    
    class Config:
        env_file = ".env"
        env_prefix = "CACHE_"


class LoggingSettings(BaseSettings):
    """Logging settings for the API."""
    
    LEVEL: str = Field(
        default_factory=lambda: os.environ.get("LOGGING_LEVEL", "INFO")
    )
    FORMAT: str = Field(
        default_factory=lambda: os.environ.get(
            "LOGGING_FORMAT", 
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    )
    FILE: Optional[str] = Field(
        default_factory=lambda: os.environ.get("LOGGING_FILE")
    )
    
    class Config:
        env_file = ".env"
        env_prefix = "LOGGING_"


class APISettings(BaseSettings):
    """API settings."""
    
    TITLE: str = Field(
        default_factory=lambda: os.environ.get("API_TITLE", "Project01 API")
    )
    DESCRIPTION: str = Field(
        default_factory=lambda: os.environ.get(
            "API_DESCRIPTION", 
            "AI-Quant-Analyst API for financial predictions and analysis"
        )
    )
    VERSION: str = Field(
        default_factory=lambda: os.environ.get("API_VERSION", "1.0.0")
    )
    PREFIX: str = Field(
        default_factory=lambda: os.environ.get("API_PREFIX", "/api")
    )
    DEBUG: bool = Field(
        default_factory=lambda: os.environ.get("API_DEBUG", "False").lower() == "true"
    )
    
    class Config:
        env_file = ".env"
        env_prefix = "API_"


class ModelSettings(BaseSettings):
    """Model settings."""
    
    MODEL_PATH: str = Field(
        default_factory=lambda: os.environ.get("MODEL_PATH", "models/baseline_xgb.pkl")
    )
    FALLBACK_MODEL: bool = Field(
        default_factory=lambda: os.environ.get("FALLBACK_MODEL", "True").lower() == "true"
    )
    
    class Config:
        env_file = ".env"
        env_prefix = "MODEL_"


class Settings(BaseSettings):
    """Main settings class that includes all other settings."""
    
    # Environment
    ENV: str = Field(
        default_factory=lambda: os.environ.get("ENV", "development")
    )
    
    # Component settings
    security: SecuritySettings = SecuritySettings()
    rate_limit: RateLimitSettings = RateLimitSettings()
    cache: CacheSettings = CacheSettings()
    logging: LoggingSettings = LoggingSettings()
    api: APISettings = APISettings()
    model: ModelSettings = ModelSettings()
    
    # Data API keys
    POLYGON_API_KEY: Optional[str] = Field(
        default_factory=lambda: os.environ.get("POLYGON_API_KEY")
    )
    NEWS_API_KEY: Optional[str] = Field(
        default_factory=lambda: os.environ.get("NEWS_API_KEY")
    )
    TWITTER_BEARER_TOKEN: Optional[str] = Field(
        default_factory=lambda: os.environ.get("TWITTER_BEARER_TOKEN")
    )
    
    class Config:
        env_file = ".env"


# Create settings instance
settings = Settings()
