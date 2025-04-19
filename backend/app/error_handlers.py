"""
Error handling module for the API.

This module provides error handling functionality for the API, including:
- Exception handlers
- Error responses
- Error logging
"""

import logging
import traceback
from typing import Dict, Any, Optional, Union, List
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from .config import settings

# Configure logging
logger = logging.getLogger(__name__)


class APIError(Exception):
    """Base API error class."""
    
    def __init__(
        self,
        status_code: int,
        detail: str,
        code: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the API error.
        
        Args:
            status_code: HTTP status code
            detail: Error detail
            code: Error code
            headers: Additional headers
        """
        self.status_code = status_code
        self.detail = detail
        self.code = code or f"error_{status_code}"
        self.headers = headers
        super().__init__(detail)


class NotFoundError(APIError):
    """Resource not found error."""
    
    def __init__(
        self,
        detail: str = "Resource not found",
        code: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the not found error.
        
        Args:
            detail: Error detail
            code: Error code
            headers: Additional headers
        """
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=detail,
            code=code or "not_found",
            headers=headers
        )


class ValidationError(APIError):
    """Validation error."""
    
    def __init__(
        self,
        detail: str = "Validation error",
        errors: Optional[List[Dict[str, Any]]] = None,
        code: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the validation error.
        
        Args:
            detail: Error detail
            errors: Validation errors
            code: Error code
            headers: Additional headers
        """
        self.errors = errors or []
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=detail,
            code=code or "validation_error",
            headers=headers
        )


class AuthenticationError(APIError):
    """Authentication error."""
    
    def __init__(
        self,
        detail: str = "Authentication error",
        code: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the authentication error.
        
        Args:
            detail: Error detail
            code: Error code
            headers: Additional headers
        """
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            code=code or "authentication_error",
            headers=headers or {"WWW-Authenticate": "Bearer"}
        )


class AuthorizationError(APIError):
    """Authorization error."""
    
    def __init__(
        self,
        detail: str = "Authorization error",
        code: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the authorization error.
        
        Args:
            detail: Error detail
            code: Error code
            headers: Additional headers
        """
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail,
            code=code or "authorization_error",
            headers=headers
        )


class RateLimitError(APIError):
    """Rate limit error."""
    
    def __init__(
        self,
        detail: str = "Rate limit exceeded",
        retry_after: Optional[int] = None,
        code: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the rate limit error.
        
        Args:
            detail: Error detail
            retry_after: Seconds to wait before retrying
            code: Error code
            headers: Additional headers
        """
        headers = headers or {}
        if retry_after:
            headers["Retry-After"] = str(retry_after)
        
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=detail,
            code=code or "rate_limit_exceeded",
            headers=headers
        )


class ServerError(APIError):
    """Server error."""
    
    def __init__(
        self,
        detail: str = "Internal server error",
        code: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the server error.
        
        Args:
            detail: Error detail
            code: Error code
            headers: Additional headers
        """
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
            code=code or "server_error",
            headers=headers
        )


def create_error_response(
    status_code: int,
    detail: str,
    code: Optional[str] = None,
    errors: Optional[List[Dict[str, Any]]] = None,
    headers: Optional[Dict[str, str]] = None
) -> JSONResponse:
    """
    Create an error response.
    
    Args:
        status_code: HTTP status code
        detail: Error detail
        code: Error code
        errors: Additional error details
        headers: Additional headers
        
    Returns:
        JSON response
    """
    content = {
        "error": {
            "code": code or f"error_{status_code}",
            "message": detail
        }
    }
    
    if errors:
        content["error"]["errors"] = errors
    
    return JSONResponse(
        status_code=status_code,
        content=content,
        headers=headers
    )


async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    """
    Handle API errors.
    
    Args:
        request: FastAPI request
        exc: API error
        
    Returns:
        JSON response
    """
    # Log error
    logger.error(
        f"API error: {exc.detail} (code: {exc.code}, status: {exc.status_code})",
        extra={"request_id": getattr(request.state, "request_id", None)}
    )
    
    # Create response
    content = {
        "error": {
            "code": exc.code,
            "message": exc.detail
        }
    }
    
    if hasattr(exc, "errors") and exc.errors:
        content["error"]["errors"] = exc.errors
    
    return JSONResponse(
        status_code=exc.status_code,
        content=content,
        headers=exc.headers
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """
    Handle HTTP exceptions.
    
    Args:
        request: FastAPI request
        exc: HTTP exception
        
    Returns:
        JSON response
    """
    # Log error
    logger.error(
        f"HTTP exception: {exc.detail} (status: {exc.status_code})",
        extra={"request_id": getattr(request.state, "request_id", None)}
    )
    
    # Create response
    return create_error_response(
        status_code=exc.status_code,
        detail=exc.detail,
        headers=exc.headers
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """
    Handle validation exceptions.
    
    Args:
        request: FastAPI request
        exc: Validation exception
        
    Returns:
        JSON response
    """
    # Log error
    logger.error(
        f"Validation error: {exc.errors()}",
        extra={"request_id": getattr(request.state, "request_id", None)}
    )
    
    # Format errors
    errors = []
    for error in exc.errors():
        errors.append({
            "loc": error["loc"],
            "msg": error["msg"],
            "type": error["type"]
        })
    
    # Create response
    return create_error_response(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail="Validation error",
        code="validation_error",
        errors=errors
    )


async def pydantic_validation_exception_handler(request: Request, exc: ValidationError) -> JSONResponse:
    """
    Handle Pydantic validation exceptions.
    
    Args:
        request: FastAPI request
        exc: Validation exception
        
    Returns:
        JSON response
    """
    # Log error
    logger.error(
        f"Pydantic validation error: {exc.errors()}",
        extra={"request_id": getattr(request.state, "request_id", None)}
    )
    
    # Format errors
    errors = []
    for error in exc.errors():
        errors.append({
            "loc": error["loc"],
            "msg": error["msg"],
            "type": error["type"]
        })
    
    # Create response
    return create_error_response(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail="Validation error",
        code="validation_error",
        errors=errors
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handle general exceptions.
    
    Args:
        request: FastAPI request
        exc: Exception
        
    Returns:
        JSON response
    """
    # Log error
    logger.error(
        f"Unhandled exception: {str(exc)}",
        exc_info=True,
        extra={"request_id": getattr(request.state, "request_id", None)}
    )
    
    # Create response
    if settings.api.DEBUG:
        # Include traceback in debug mode
        return create_error_response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
            code="server_error",
            errors=[{
                "traceback": traceback.format_exc()
            }]
        )
    else:
        # Generic error in production
        return create_error_response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
            code="server_error"
        )
