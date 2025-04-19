"""Main API module.

This module provides the main FastAPI application and routes.
"""

import time
from datetime import timedelta
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, Depends, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import ValidationError

# Import original prediction function for backward compatibility
from predict import run_prediction

# Try to import advanced prediction module
try:
    from .advanced_predict import run_advanced_prediction
    ADVANCED_PREDICTION_AVAILABLE = True
except ImportError:
    ADVANCED_PREDICTION_AVAILABLE = False

# Import new modules
try:
    from .config import settings
    from .auth import (
        Token, User, authenticate_user, create_access_token,
        get_current_user, get_api_key, get_current_active_user
    )
    from .models import (
        PredictionRequest, PredictionResponse, ModelInfoResponse,
        model_manager, predict
    )
    from .rate_limit import rate_limit_middleware
    from .cache import cache, cache_middleware
    from .logging_config import RequestLogMiddleware
    from .error_handlers import (
        APIError, http_exception_handler, validation_exception_handler,
        api_error_handler, general_exception_handler
    )

    # If imports succeed, we're using the new structure
    NEW_API_STRUCTURE = True
except ImportError:
    # If imports fail, we're using the old structure
    NEW_API_STRUCTURE = False
    # Import the old models
    from models import PredictionRequest, PredictionResponse

# Create FastAPI app
if NEW_API_STRUCTURE:
    app = FastAPI(
        title=settings.api.TITLE,
        description=settings.api.DESCRIPTION,
        version=settings.api.VERSION,
        docs_url=None,  # Disable default docs
        redoc_url=None,  # Disable default redoc
        openapi_url=f"{settings.api.PREFIX}/openapi.json" if not settings.ENV == "production" else None
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.security.CORS_ORIGINS,
        allow_credentials=settings.security.CORS_ALLOW_CREDENTIALS,
        allow_methods=settings.security.CORS_ALLOW_METHODS,
        allow_headers=settings.security.CORS_ALLOW_HEADERS,
    )

    # Add request logging middleware
    app.add_middleware(RequestLogMiddleware)

    # Add rate limiting middleware
    @app.middleware("http")
    async def rate_limiting(request: Request, call_next):
        return await rate_limit_middleware(request, call_next)

    # Add caching middleware
    @app.middleware("http")
    async def caching(request: Request, call_next):
        return await cache_middleware(request, call_next)

    # Add exception handlers
    app.add_exception_handler(APIError, api_error_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(ValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)

    # Mount static files
    try:
        app.mount("/static", StaticFiles(directory="static"), name="static")
    except RuntimeError:
        pass  # Static directory doesn't exist

    # Custom OpenAPI schema
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema

        openapi_schema = get_openapi(
            title=settings.api.TITLE,
            version=settings.api.VERSION,
            description=settings.api.DESCRIPTION,
            routes=app.routes,
        )

        # Add API key security scheme
        openapi_schema["components"]["securitySchemes"] = {
            "ApiKeyAuth": {
                "type": "apiKey",
                "in": "header",
                "name": settings.security.API_KEY_HEADER
            },
            "OAuth2PasswordBearer": {
                "type": "oauth2",
                "flows": {
                    "password": {
                        "tokenUrl": f"{settings.api.PREFIX}/token",
                        "scopes": {}
                    }
                }
            }
        }

        # Add security requirement to all operations
        for path in openapi_schema["paths"].values():
            for operation in path.values():
                operation["security"] = [
                    {"ApiKeyAuth": []},
                    {"OAuth2PasswordBearer": []}
                ]

        app.openapi_schema = openapi_schema
        return app.openapi_schema

    app.openapi = custom_openapi

    # Custom docs endpoints
    @app.get("/docs", include_in_schema=False)
    async def custom_swagger_ui_html():
        return get_swagger_ui_html(
            openapi_url=f"{settings.api.PREFIX}/openapi.json",
            title=f"{settings.api.TITLE} - Swagger UI",
            oauth2_redirect_url=f"{settings.api.PREFIX}/docs/oauth2-redirect",
            swagger_js_url="/static/swagger-ui-bundle.js",
            swagger_css_url="/static/swagger-ui.css",
        )

    # Root endpoint
    @app.get("/")
    async def root():
        return {
            "name": settings.api.TITLE,
            "version": settings.api.VERSION,
            "description": settings.api.DESCRIPTION,
            "docs_url": f"{settings.api.PREFIX}/docs"
        }

    # Health check endpoint
    @app.get(f"{settings.api.PREFIX}/health")
    async def health_check():
        return {
            "status": "ok",
            "timestamp": time.time(),
            "version": settings.api.VERSION,
            "environment": settings.ENV
        }

    # Authentication endpoints
    @app.post(f"{settings.api.PREFIX}/token", response_model=Token)
    async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
        user = authenticate_user(form_data.username, form_data.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Create access token
        access_token_expires = timedelta(minutes=settings.security.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.username, "scopes": user.scopes},
            expires_delta=access_token_expires
        )

        # Return token
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_at": int(time.time() + settings.security.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60)
        }

    @app.get(f"{settings.api.PREFIX}/users/me", response_model=User)
    async def read_users_me(current_user: User = Depends(get_current_active_user)):
        return current_user

    # Model endpoints
    @app.get(f"{settings.api.PREFIX}/model/info", response_model=ModelInfoResponse)
    async def get_model_info(api_key: str = Depends(get_api_key)):
        """Get information about the loaded model."""
        return model_manager.get_model_info()

    # New prediction endpoint with authentication
    @app.post(f"{settings.api.PREFIX}/predict/v2", response_model=PredictionResponse)
    async def make_prediction(request: PredictionRequest, api_key: str = Depends(get_api_key)):
        """Make a prediction using the loaded model."""
        # Convert request to dict
        data = request.dict()

        # Make prediction
        result = await predict(data)

        # Check for error
        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"]
            )

        # Create response
        response = {
            "timestamps": ["2023-01-01", "2023-01-02", "2023-01-03"],  # Placeholder
            "prices": [100.0, 101.0, 102.0],  # Placeholder
            "ma_5": [100.0, 100.5, 101.0],  # Placeholder
            "rsi_14": [50.0, 55.0, 60.0],  # Placeholder
            "sentiment": {"positive": 0.6, "negative": 0.2, "neutral": 0.2},  # Placeholder
            "prediction": result["prediction"]
        }

        # Add confidence interval if available
        if "confidence_interval" in result:
            response["confidence_interval"] = result["confidence_interval"]

        # Add feature importance if available
        if "feature_importance" in result:
            response["feature_importance"] = result["feature_importance"]

        return response

    # Advanced prediction endpoints
    if ADVANCED_PREDICTION_AVAILABLE:
        from pydantic import BaseModel, Field

        class AdvancedPredictionRequest(BaseModel):
            """Model for advanced prediction request."""
            symbol: str = Field(..., description="Stock ticker symbol (e.g., AAPL)")
            days: int = Field(30, description="Number of days of historical data to use", ge=1, le=365)

        class AdvancedPredictionResponse(BaseModel):
            """Model for advanced prediction response."""
            symbol: str = Field(..., description="Stock ticker symbol")
            latest_price: float = Field(..., description="Latest price")
            predicted_return: float = Field(..., description="Predicted return")
            predicted_price: float = Field(..., description="Predicted price")
            confidence_interval: Optional[Dict[str, float]] = Field(None, description="Confidence interval for prediction")
            uncertainty: Optional[float] = Field(None, description="Uncertainty of prediction")
            sentiment: Dict[str, float] = Field(..., description="Sentiment analysis results")
            historical_data: Dict[str, List[Any]] = Field(..., description="Historical data")

        @app.post(f"{settings.api.PREFIX}/predict/advanced", response_model=AdvancedPredictionResponse)
        async def advanced_prediction(request: AdvancedPredictionRequest, api_key: str = Depends(get_api_key)):
            """Make an advanced prediction using ensemble models, TFT, and FinBERT."""
            try:
                result = run_advanced_prediction(request.symbol, request.days)
                return result
            except ValueError as e:
                # This indicates we couldn't get real data
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=str(e)
                )
            except Exception as e:
                # Other errors
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Prediction failed: {str(e)}"
                )

    # Cache endpoints
    @app.get(f"{settings.api.PREFIX}/cache/stats")
    async def get_cache_stats(api_key: str = Depends(get_api_key)):
        """Get cache statistics."""
        return await cache.get_stats()

    @app.post(f"{settings.api.PREFIX}/cache/clear")
    async def clear_cache(api_key: str = Depends(get_api_key)):
        """Clear the cache."""
        await cache.clear()
        return {"status": "ok", "message": "Cache cleared"}

    # Startup event
    @app.on_event("startup")
    async def startup_event():
        # Log startup
        import logging
        logger = logging.getLogger("api")
        logger.info(f"Starting {settings.api.TITLE} v{settings.api.VERSION}")
        logger.info(f"Environment: {settings.ENV}")

        # Load model
        model_manager.load_model()

    # Shutdown event
    @app.on_event("shutdown")
    async def shutdown_event():
        # Log shutdown
        import logging
        logger = logging.getLogger("api")
        logger.info(f"Shutting down {settings.api.TITLE} v{settings.api.VERSION}")

else:
    # Legacy app setup
    app = FastAPI(title="Project01 API")

    # Add CORS middleware to allow frontend to call the API
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, replace with specific origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Legacy endpoints (always available)
@app.get("/api/health")
def health_check():
    """Health check endpoint"""
    return {"status": "ok"}

@app.post("/api/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    """Run prediction for a given stock symbol"""
    try:
        return run_prediction(payload.symbol, payload.days)
    except ValueError as e:
        # This indicates we couldn't get real data
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        # Other errors
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True if not NEW_API_STRUCTURE else settings.api.DEBUG
    )
