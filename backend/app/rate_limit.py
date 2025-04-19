"""
Rate limiting module for the API.

This module provides rate limiting functionality for the API, including:
- Fixed window rate limiting
- Sliding window rate limiting
- Token bucket rate limiting
"""

import time
from datetime import datetime
from typing import Dict, Optional, Tuple, Callable, Any, Union
import asyncio
import logging
from fastapi import Request, HTTPException, status

from .config import settings

# Configure logging
logger = logging.getLogger(__name__)


class RateLimiter:
    """Base rate limiter class."""
    
    def __init__(self, limit: int, window: int = 60):
        """
        Initialize the rate limiter.
        
        Args:
            limit: Maximum number of requests per window
            window: Window size in seconds
        """
        self.limit = limit
        self.window = window
        self.store: Dict[str, Dict[str, Any]] = {}
    
    async def is_rate_limited(self, key: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if a key is rate limited.
        
        Args:
            key: Key to check
            
        Returns:
            Tuple of (is_limited, rate_limit_info)
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_window_stats(self, key: str) -> Dict[str, Any]:
        """
        Get window statistics for a key.
        
        Args:
            key: Key to get statistics for
            
        Returns:
            Dictionary with window statistics
        """
        raise NotImplementedError("Subclasses must implement this method")


class FixedWindowRateLimiter(RateLimiter):
    """Fixed window rate limiter."""
    
    async def is_rate_limited(self, key: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if a key is rate limited using a fixed window.
        
        Args:
            key: Key to check
            
        Returns:
            Tuple of (is_limited, rate_limit_info)
        """
        now = int(time.time())
        window_start = now - (now % self.window)
        window_key = f"{key}:{window_start}"
        
        # Get or create window data
        if window_key not in self.store:
            self.store[window_key] = {
                "count": 0,
                "window_start": window_start,
                "window_end": window_start + self.window,
                "reset_at": window_start + self.window
            }
        
        # Clean up old windows
        for k in list(self.store.keys()):
            if k.startswith(f"{key}:") and self.store[k]["window_end"] < now:
                del self.store[k]
        
        # Increment counter
        self.store[window_key]["count"] += 1
        
        # Check if rate limited
        is_limited = self.store[window_key]["count"] > self.limit
        
        # Prepare rate limit info
        rate_limit_info = {
            "limit": self.limit,
            "remaining": max(0, self.limit - self.store[window_key]["count"]),
            "reset": self.store[window_key]["reset_at"],
            "window": self.window
        }
        
        return is_limited, rate_limit_info
    
    def get_window_stats(self, key: str) -> Dict[str, Any]:
        """
        Get window statistics for a key.
        
        Args:
            key: Key to get statistics for
            
        Returns:
            Dictionary with window statistics
        """
        now = int(time.time())
        window_start = now - (now % self.window)
        window_key = f"{key}:{window_start}"
        
        if window_key not in self.store:
            return {
                "count": 0,
                "window_start": window_start,
                "window_end": window_start + self.window,
                "reset_at": window_start + self.window,
                "limit": self.limit,
                "remaining": self.limit
            }
        
        return {
            "count": self.store[window_key]["count"],
            "window_start": self.store[window_key]["window_start"],
            "window_end": self.store[window_key]["window_end"],
            "reset_at": self.store[window_key]["reset_at"],
            "limit": self.limit,
            "remaining": max(0, self.limit - self.store[window_key]["count"])
        }


class SlidingWindowRateLimiter(RateLimiter):
    """Sliding window rate limiter."""
    
    async def is_rate_limited(self, key: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if a key is rate limited using a sliding window.
        
        Args:
            key: Key to check
            
        Returns:
            Tuple of (is_limited, rate_limit_info)
        """
        now = time.time()
        window_start = now - self.window
        
        # Initialize key data if not exists
        if key not in self.store:
            self.store[key] = {
                "requests": [],
                "count": 0
            }
        
        # Remove old requests
        self.store[key]["requests"] = [
            r for r in self.store[key]["requests"] if r > window_start
        ]
        
        # Update count
        self.store[key]["count"] = len(self.store[key]["requests"])
        
        # Add current request
        self.store[key]["requests"].append(now)
        self.store[key]["count"] += 1
        
        # Check if rate limited
        is_limited = self.store[key]["count"] > self.limit
        
        # Calculate reset time
        reset_at = self.store[key]["requests"][0] + self.window if self.store[key]["requests"] else now + self.window
        
        # Prepare rate limit info
        rate_limit_info = {
            "limit": self.limit,
            "remaining": max(0, self.limit - self.store[key]["count"]),
            "reset": reset_at,
            "window": self.window
        }
        
        return is_limited, rate_limit_info
    
    def get_window_stats(self, key: str) -> Dict[str, Any]:
        """
        Get window statistics for a key.
        
        Args:
            key: Key to get statistics for
            
        Returns:
            Dictionary with window statistics
        """
        now = time.time()
        window_start = now - self.window
        
        if key not in self.store:
            return {
                "count": 0,
                "window_start": window_start,
                "window_end": now,
                "reset_at": now + self.window,
                "limit": self.limit,
                "remaining": self.limit
            }
        
        # Remove old requests
        self.store[key]["requests"] = [
            r for r in self.store[key]["requests"] if r > window_start
        ]
        
        # Update count
        self.store[key]["count"] = len(self.store[key]["requests"])
        
        # Calculate reset time
        reset_at = self.store[key]["requests"][0] + self.window if self.store[key]["requests"] else now + self.window
        
        return {
            "count": self.store[key]["count"],
            "window_start": window_start,
            "window_end": now,
            "reset_at": reset_at,
            "limit": self.limit,
            "remaining": max(0, self.limit - self.store[key]["count"])
        }


class TokenBucketRateLimiter(RateLimiter):
    """Token bucket rate limiter."""
    
    def __init__(self, limit: int, window: int = 60, refill_rate: Optional[float] = None):
        """
        Initialize the token bucket rate limiter.
        
        Args:
            limit: Maximum number of tokens in the bucket
            window: Window size in seconds
            refill_rate: Rate at which tokens are refilled (tokens per second)
        """
        super().__init__(limit, window)
        self.refill_rate = refill_rate or (limit / window)
    
    async def is_rate_limited(self, key: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if a key is rate limited using a token bucket.
        
        Args:
            key: Key to check
            
        Returns:
            Tuple of (is_limited, rate_limit_info)
        """
        now = time.time()
        
        # Initialize key data if not exists
        if key not in self.store:
            self.store[key] = {
                "tokens": self.limit,
                "last_refill": now
            }
        
        # Refill tokens
        time_passed = now - self.store[key]["last_refill"]
        tokens_to_add = time_passed * self.refill_rate
        self.store[key]["tokens"] = min(self.limit, self.store[key]["tokens"] + tokens_to_add)
        self.store[key]["last_refill"] = now
        
        # Check if we have enough tokens
        if self.store[key]["tokens"] < 1:
            # Calculate time until next token
            time_until_next_token = (1 - self.store[key]["tokens"]) / self.refill_rate
            reset_at = now + time_until_next_token
            
            # Prepare rate limit info
            rate_limit_info = {
                "limit": self.limit,
                "remaining": 0,
                "reset": reset_at,
                "window": self.window
            }
            
            return True, rate_limit_info
        
        # Consume a token
        self.store[key]["tokens"] -= 1
        
        # Calculate time until bucket is full
        time_until_full = (self.limit - self.store[key]["tokens"]) / self.refill_rate
        reset_at = now + time_until_full
        
        # Prepare rate limit info
        rate_limit_info = {
            "limit": self.limit,
            "remaining": int(self.store[key]["tokens"]),
            "reset": reset_at,
            "window": self.window
        }
        
        return False, rate_limit_info
    
    def get_window_stats(self, key: str) -> Dict[str, Any]:
        """
        Get window statistics for a key.
        
        Args:
            key: Key to get statistics for
            
        Returns:
            Dictionary with window statistics
        """
        now = time.time()
        
        if key not in self.store:
            return {
                "tokens": self.limit,
                "last_refill": now,
                "limit": self.limit,
                "remaining": self.limit,
                "reset_at": now
            }
        
        # Refill tokens
        time_passed = now - self.store[key]["last_refill"]
        tokens_to_add = time_passed * self.refill_rate
        tokens = min(self.limit, self.store[key]["tokens"] + tokens_to_add)
        
        # Calculate time until bucket is full
        time_until_full = (self.limit - tokens) / self.refill_rate
        reset_at = now + time_until_full
        
        return {
            "tokens": tokens,
            "last_refill": self.store[key]["last_refill"],
            "limit": self.limit,
            "remaining": int(tokens),
            "reset_at": reset_at
        }


# Factory function to create rate limiters
def create_rate_limiter(strategy: str = "fixed-window", limit: int = 60, window: int = 60) -> RateLimiter:
    """
    Create a rate limiter.
    
    Args:
        strategy: Rate limiting strategy
        limit: Maximum number of requests per window
        window: Window size in seconds
        
    Returns:
        Rate limiter instance
    """
    if strategy == "sliding-window":
        return SlidingWindowRateLimiter(limit, window)
    elif strategy == "token-bucket":
        return TokenBucketRateLimiter(limit, window)
    else:
        return FixedWindowRateLimiter(limit, window)


# Create rate limiter instance
rate_limiter = create_rate_limiter(
    strategy=settings.rate_limit.RATE_LIMIT_STRATEGY,
    limit=settings.rate_limit.RATE_LIMIT_PER_MINUTE,
    window=60
)


# Rate limiting middleware
async def rate_limit_middleware(request: Request, call_next: Callable) -> Any:
    """
    Rate limiting middleware.
    
    Args:
        request: FastAPI request
        call_next: Next middleware or endpoint
        
    Returns:
        Response
    """
    if not settings.rate_limit.ENABLED:
        return await call_next(request)
    
    # Get client identifier (IP address or API key)
    client_id = request.client.host
    
    # Check for API key in headers
    api_key = request.headers.get(settings.security.API_KEY_HEADER)
    if api_key:
        client_id = f"api_key:{api_key}"
    
    # Check rate limit
    is_limited, rate_limit_info = await rate_limiter.is_rate_limited(client_id)
    
    # Add rate limit headers to response
    response = await call_next(request)
    response.headers["X-RateLimit-Limit"] = str(rate_limit_info["limit"])
    response.headers["X-RateLimit-Remaining"] = str(rate_limit_info["remaining"])
    response.headers["X-RateLimit-Reset"] = str(int(rate_limit_info["reset"]))
    
    # If rate limited, return 429 Too Many Requests
    if is_limited:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={"Retry-After": str(int(rate_limit_info["reset"] - time.time()))}
        )
    
    return response
