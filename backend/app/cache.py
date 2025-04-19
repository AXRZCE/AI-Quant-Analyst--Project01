"""
Caching module for the API.

This module provides caching functionality for the API, including:
- In-memory caching
- Redis caching
- Cache key generation
- Cache invalidation
"""

import time
import json
import hashlib
import logging
from typing import Dict, Any, Optional, Callable, Union, TypeVar, Generic, List
from functools import wraps
from datetime import datetime, timedelta
from fastapi import Request, Response

from .config import settings

# Configure logging
logger = logging.getLogger(__name__)

# Type variable for cache value
T = TypeVar('T')


class Cache(Generic[T]):
    """Base cache class."""
    
    def __init__(self, ttl: int = 300):
        """
        Initialize the cache.
        
        Args:
            ttl: Time to live in seconds
        """
        self.ttl = ttl
    
    async def get(self, key: str) -> Optional[T]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    async def set(self, key: str, value: T, ttl: Optional[int] = None) -> None:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (overrides default)
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    async def delete(self, key: str) -> None:
        """
        Delete a value from the cache.
        
        Args:
            key: Cache key
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    async def clear(self) -> None:
        """Clear the cache."""
        raise NotImplementedError("Subclasses must implement this method")
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        raise NotImplementedError("Subclasses must implement this method")


class MemoryCache(Cache[T]):
    """In-memory cache."""
    
    def __init__(self, ttl: int = 300, max_size: int = 1000):
        """
        Initialize the in-memory cache.
        
        Args:
            ttl: Time to live in seconds
            max_size: Maximum number of items in the cache
        """
        super().__init__(ttl)
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    async def get(self, key: str) -> Optional[T]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        # Clean up expired items
        await self._cleanup()
        
        # Check if key exists and is not expired
        if key in self.cache and self.cache[key]["expires_at"] > time.time():
            self.hits += 1
            return self.cache[key]["value"]
        
        # Key not found or expired
        self.misses += 1
        return None
    
    async def set(self, key: str, value: T, ttl: Optional[int] = None) -> None:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (overrides default)
        """
        # Clean up expired items
        await self._cleanup()
        
        # Check if cache is full
        if len(self.cache) >= self.max_size and key not in self.cache:
            # Remove oldest item
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]["created_at"])
            del self.cache[oldest_key]
        
        # Set value
        self.cache[key] = {
            "value": value,
            "created_at": time.time(),
            "expires_at": time.time() + (ttl if ttl is not None else self.ttl)
        }
    
    async def delete(self, key: str) -> None:
        """
        Delete a value from the cache.
        
        Args:
            key: Cache key
        """
        if key in self.cache:
            del self.cache[key]
    
    async def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_ratio": self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
        }
    
    async def _cleanup(self) -> None:
        """Clean up expired items."""
        now = time.time()
        expired_keys = [k for k, v in self.cache.items() if v["expires_at"] <= now]
        for key in expired_keys:
            del self.cache[key]


class RedisCache(Cache[T]):
    """Redis cache."""
    
    def __init__(self, ttl: int = 300, redis_url: Optional[str] = None):
        """
        Initialize the Redis cache.
        
        Args:
            ttl: Time to live in seconds
            redis_url: Redis URL
        """
        super().__init__(ttl)
        self.redis_url = redis_url or settings.cache.REDIS_URL
        self.redis = None
        self.hits = 0
        self.misses = 0
        self._connect()
    
    def _connect(self) -> None:
        """Connect to Redis."""
        try:
            import redis
            self.redis = redis.from_url(self.redis_url)
            logger.info(f"Connected to Redis at {self.redis_url}")
        except ImportError:
            logger.error("Redis package not installed. Install with 'pip install redis'")
            self.redis = None
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis = None
    
    async def get(self, key: str) -> Optional[T]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        if not self.redis:
            return None
        
        try:
            value = self.redis.get(key)
            if value:
                self.hits += 1
                return json.loads(value)
            
            self.misses += 1
            return None
        except Exception as e:
            logger.error(f"Error getting value from Redis: {e}")
            return None
    
    async def set(self, key: str, value: T, ttl: Optional[int] = None) -> None:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (overrides default)
        """
        if not self.redis:
            return
        
        try:
            self.redis.setex(
                key,
                ttl if ttl is not None else self.ttl,
                json.dumps(value)
            )
        except Exception as e:
            logger.error(f"Error setting value in Redis: {e}")
    
    async def delete(self, key: str) -> None:
        """
        Delete a value from the cache.
        
        Args:
            key: Cache key
        """
        if not self.redis:
            return
        
        try:
            self.redis.delete(key)
        except Exception as e:
            logger.error(f"Error deleting value from Redis: {e}")
    
    async def clear(self) -> None:
        """Clear the cache."""
        if not self.redis:
            return
        
        try:
            self.redis.flushdb()
            self.hits = 0
            self.misses = 0
        except Exception as e:
            logger.error(f"Error clearing Redis cache: {e}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        if not self.redis:
            return {
                "connected": False,
                "hits": self.hits,
                "misses": self.misses,
                "hit_ratio": 0
            }
        
        try:
            info = self.redis.info()
            return {
                "connected": True,
                "hits": self.hits,
                "misses": self.misses,
                "hit_ratio": self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0,
                "memory_used": info.get("used_memory_human", "N/A"),
                "clients_connected": info.get("connected_clients", "N/A"),
                "uptime": info.get("uptime_in_seconds", "N/A")
            }
        except Exception as e:
            logger.error(f"Error getting Redis stats: {e}")
            return {
                "connected": False,
                "error": str(e),
                "hits": self.hits,
                "misses": self.misses,
                "hit_ratio": self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
            }


# Factory function to create cache
def create_cache(backend: str = "memory", ttl: int = 300, **kwargs) -> Cache:
    """
    Create a cache.
    
    Args:
        backend: Cache backend
        ttl: Time to live in seconds
        **kwargs: Additional arguments for the cache
        
    Returns:
        Cache instance
    """
    if backend == "redis" and settings.cache.REDIS_URL:
        return RedisCache(ttl, settings.cache.REDIS_URL)
    else:
        return MemoryCache(ttl, settings.cache.MAX_SIZE)


# Create cache instance
cache = create_cache(
    backend=settings.cache.BACKEND,
    ttl=settings.cache.TTL_SECONDS
)


# Cache key generation
def generate_cache_key(prefix: str, *args, **kwargs) -> str:
    """
    Generate a cache key.
    
    Args:
        prefix: Key prefix
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        Cache key
    """
    # Convert args and kwargs to strings
    args_str = ",".join(str(arg) for arg in args)
    kwargs_str = ",".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
    
    # Combine and hash
    key_str = f"{prefix}:{args_str}:{kwargs_str}"
    key_hash = hashlib.md5(key_str.encode()).hexdigest()
    
    return f"{prefix}:{key_hash}"


# Cache decorator
def cached(prefix: str, ttl: Optional[int] = None):
    """
    Cache decorator.
    
    Args:
        prefix: Cache key prefix
        ttl: Time to live in seconds
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if not settings.cache.ENABLED:
                return await func(*args, **kwargs)
            
            # Generate cache key
            cache_key = generate_cache_key(prefix, *args, **kwargs)
            
            # Try to get from cache
            cached_value = await cache.get(cache_key)
            if cached_value is not None:
                return cached_value
            
            # Call function
            result = await func(*args, **kwargs)
            
            # Cache result
            await cache.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator


# Request caching middleware
async def cache_middleware(request: Request, call_next: Callable) -> Response:
    """
    Cache middleware for FastAPI.
    
    Args:
        request: FastAPI request
        call_next: Next middleware or endpoint
        
    Returns:
        Response
    """
    if not settings.cache.ENABLED:
        return await call_next(request)
    
    # Only cache GET requests
    if request.method != "GET":
        return await call_next(request)
    
    # Generate cache key
    cache_key = generate_cache_key(
        "http",
        request.url.path,
        str(request.query_params)
    )
    
    # Try to get from cache
    cached_response = await cache.get(cache_key)
    if cached_response is not None:
        # Reconstruct response
        response = Response(
            content=cached_response["content"],
            status_code=cached_response["status_code"],
            headers=cached_response["headers"],
            media_type=cached_response["media_type"]
        )
        response.headers["X-Cache"] = "HIT"
        return response
    
    # Call next middleware or endpoint
    response = await call_next(request)
    
    # Cache response if successful
    if 200 <= response.status_code < 400:
        # Get response content
        response_body = b""
        async for chunk in response.body_iterator:
            response_body += chunk
        
        # Cache response
        await cache.set(
            cache_key,
            {
                "content": response_body,
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "media_type": response.media_type
            },
            ttl=settings.cache.TTL_SECONDS
        )
        
        # Create new response with the same content
        new_response = Response(
            content=response_body,
            status_code=response.status_code,
            headers=response.headers,
            media_type=response.media_type
        )
        new_response.headers["X-Cache"] = "MISS"
        return new_response
    
    return response
