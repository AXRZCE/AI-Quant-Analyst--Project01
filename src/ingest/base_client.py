"""
Base API client with retry logic and error handling.

This module provides a base class for API clients with common functionality
such as retry logic, error handling, and logging.
"""

import time
import logging
import functools
from typing import Any, Callable, Dict, Optional, TypeVar, cast
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logger = logging.getLogger(__name__)

# Type variable for return type of decorated functions
T = TypeVar('T')

class BaseAPIClient:
    """Base class for API clients with retry logic and error handling."""
    
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        status_forcelist: tuple = (500, 502, 503, 504),
    ):
        """
        Initialize the base API client.
        
        Args:
            base_url: Base URL for the API
            api_key: API key for authentication
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            backoff_factor: Backoff factor for retries
            status_forcelist: HTTP status codes to retry on
        """
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        
        # Create a session with retry logic
        self.session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=status_forcelist,
            allowed_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
        )
        
        # Mount the adapter to the session
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    def with_retry(
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        exceptions: tuple = (requests.RequestException, ConnectionError, TimeoutError),
    ) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """
        Decorator for retrying functions on failure.
        
        Args:
            max_retries: Maximum number of retries
            backoff_factor: Backoff factor for retries
            exceptions: Exceptions to catch and retry on
            
        Returns:
            Decorated function with retry logic
        """
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> T:
                last_exception = None
                for attempt in range(max_retries + 1):
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e
                        if attempt < max_retries:
                            sleep_time = backoff_factor * (2 ** attempt)
                            logger.warning(
                                f"Attempt {attempt + 1}/{max_retries + 1} failed: {str(e)}. "
                                f"Retrying in {sleep_time:.2f} seconds..."
                            )
                            time.sleep(sleep_time)
                        else:
                            logger.error(f"All {max_retries + 1} attempts failed.")
                            raise
                
                # This should never be reached, but just in case
                if last_exception:
                    raise last_exception
                return cast(T, None)  # This will never be reached
            
            return wrapper
        
        return decorator
    
    def handle_response(self, response: requests.Response) -> Dict[str, Any]:
        """
        Handle API response and check for errors.
        
        Args:
            response: Response object from requests
            
        Returns:
            Parsed JSON response
            
        Raises:
            requests.HTTPError: If the response contains an error
        """
        # Check for HTTP errors
        response.raise_for_status()
        
        # Parse JSON response
        try:
            return response.json()
        except ValueError:
            logger.error(f"Failed to parse JSON response: {response.text}")
            raise ValueError(f"Invalid JSON response: {response.text}")
    
    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make a GET request to the API.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            
        Returns:
            Parsed JSON response
        """
        url = f"{self.base_url}{endpoint}"
        response = self.session.get(url, params=params, timeout=self.timeout)
        return self.handle_response(response)
    
    def post(self, endpoint: str, data: Optional[Dict[str, Any]] = None, json: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make a POST request to the API.
        
        Args:
            endpoint: API endpoint
            data: Form data
            json: JSON data
            
        Returns:
            Parsed JSON response
        """
        url = f"{self.base_url}{endpoint}"
        response = self.session.post(url, data=data, json=json, timeout=self.timeout)
        return self.handle_response(response)
