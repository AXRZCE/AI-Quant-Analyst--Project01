"""
Logging configuration for the API.

This module provides logging configuration for the API, including:
- Console logging
- File logging
- Request logging
- Error logging
"""

import logging
import sys
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional, Callable
from pathlib import Path
from fastapi import Request, Response
import uuid

from .config import settings

# Create logs directory if it doesn't exist
if settings.logging.FILE:
    log_dir = Path(settings.logging.FILE).parent
    log_dir.mkdir(parents=True, exist_ok=True)


def setup_logging() -> None:
    """Set up logging configuration."""
    # Get log level
    log_level = getattr(logging, settings.logging.LEVEL.upper(), logging.INFO)
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(settings.logging.FORMAT)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if file is specified
    if settings.logging.FILE:
        file_handler = logging.FileHandler(settings.logging.FILE)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Set log level for other loggers
    logging.getLogger("uvicorn").setLevel(log_level)
    logging.getLogger("uvicorn.access").setLevel(log_level)
    
    # Log configuration
    logger.info(f"Logging configured with level {settings.logging.LEVEL}")
    if settings.logging.FILE:
        logger.info(f"Logging to file {settings.logging.FILE}")


class RequestLogMiddleware:
    """Middleware for logging requests and responses."""
    
    def __init__(self, app):
        """
        Initialize the middleware.
        
        Args:
            app: FastAPI application
        """
        self.app = app
        self.logger = logging.getLogger("api.request")
    
    async def __call__(self, request: Request, call_next: Callable) -> Response:
        """
        Process the request and log it.
        
        Args:
            request: FastAPI request
            call_next: Next middleware or endpoint
            
        Returns:
            Response
        """
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Get API key if present
        api_key = request.headers.get(settings.security.API_KEY_HEADER, "none")
        api_key_masked = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "none"
        
        # Log request
        start_time = time.time()
        self.logger.info(
            f"Request {request_id}: {request.method} {request.url.path} "
            f"from {client_ip} with API key {api_key_masked}"
        )
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            # Log response
            self.logger.info(
                f"Response {request_id}: {response.status_code} in {process_time:.3f}s"
            )
            
            return response
        except Exception as e:
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Log error
            self.logger.error(
                f"Error {request_id}: {str(e)} in {process_time:.3f}s",
                exc_info=True
            )
            
            # Re-raise exception
            raise


class JSONLogFormatter(logging.Formatter):
    """JSON log formatter."""
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record as JSON.
        
        Args:
            record: Log record
            
        Returns:
            JSON string
        """
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info)
            }
        
        # Add extra fields if present
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id
        
        if hasattr(record, "user"):
            log_data["user"] = record.user
        
        return json.dumps(log_data)


def setup_json_logging() -> None:
    """Set up JSON logging."""
    # Get log level
    log_level = getattr(logging, settings.logging.LEVEL.upper(), logging.INFO)
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = JSONLogFormatter()
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if file is specified
    if settings.logging.FILE:
        file_handler = logging.FileHandler(settings.logging.FILE)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Set log level for other loggers
    logging.getLogger("uvicorn").setLevel(log_level)
    logging.getLogger("uvicorn.access").setLevel(log_level)
    
    # Log configuration
    logger.info(f"JSON logging configured with level {settings.logging.LEVEL}")
    if settings.logging.FILE:
        logger.info(f"Logging to file {settings.logging.FILE}")


# Set up logging based on environment
if settings.ENV == "production":
    setup_json_logging()
else:
    setup_logging()
