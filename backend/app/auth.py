"""
Authentication module for the API.

This module provides authentication functionality for the API, including:
- API key authentication
- JWT token authentication
- User management
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader, OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field

from .config import settings

# API Key security scheme
api_key_header = APIKeyHeader(name=settings.security.API_KEY_HEADER, auto_error=False)

# OAuth2 security scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class Token(BaseModel):
    """Token model."""
    access_token: str
    token_type: str
    expires_at: int


class TokenData(BaseModel):
    """Token data model."""
    username: Optional[str] = None
    scopes: List[str] = []


class User(BaseModel):
    """User model."""
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: bool = False
    scopes: List[str] = []


class UserInDB(User):
    """User model with hashed password."""
    hashed_password: str


# Mock user database - in production, use a real database
USERS_DB = {
    "admin": {
        "username": "admin",
        "full_name": "Administrator",
        "email": "admin@example.com",
        "hashed_password": pwd_context.hash("admin"),
        "disabled": False,
        "scopes": ["admin", "read", "write"]
    },
    "user": {
        "username": "user",
        "full_name": "Regular User",
        "email": "user@example.com",
        "hashed_password": pwd_context.hash("user"),
        "disabled": False,
        "scopes": ["read"]
    }
}


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Get password hash."""
    return pwd_context.hash(password)


def get_user(username: str) -> Optional[UserInDB]:
    """Get user from database."""
    if username in USERS_DB:
        user_dict = USERS_DB[username]
        return UserInDB(**user_dict)
    return None


def authenticate_user(username: str, password: str) -> Optional[User]:
    """Authenticate a user."""
    user = get_user(username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create an access token."""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.security.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode, 
        settings.security.JWT_SECRET_KEY, 
        algorithm=settings.security.JWT_ALGORITHM
    )
    
    return encoded_jwt


async def get_current_user_from_token(token: str = Depends(oauth2_scheme)) -> User:
    """Get current user from token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    if not token:
        raise credentials_exception
    
    try:
        payload = jwt.decode(
            token, 
            settings.security.JWT_SECRET_KEY, 
            algorithms=[settings.security.JWT_ALGORITHM]
        )
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        
        token_scopes = payload.get("scopes", [])
        token_data = TokenData(username=username, scopes=token_scopes)
    except JWTError:
        raise credentials_exception
    
    user = get_user(token_data.username)
    if user is None:
        raise credentials_exception
    
    return user


async def get_current_active_user(current_user: User = Depends(get_current_user_from_token)) -> User:
    """Get current active user."""
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


async def validate_api_key(api_key: str = Security(api_key_header)) -> bool:
    """Validate API key."""
    if not api_key:
        return False
    
    if api_key in settings.security.API_KEYS:
        return True
    
    return False


async def get_api_key(api_key: str = Security(api_key_header)) -> str:
    """Get and validate API key."""
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key is missing",
            headers={"WWW-Authenticate": settings.security.API_KEY_HEADER},
        )
    
    if api_key not in settings.security.API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": settings.security.API_KEY_HEADER},
        )
    
    return api_key


# Combined authentication - either API key or JWT token
async def get_current_user(
    api_key: Optional[str] = Security(api_key_header),
    token: Optional[str] = Depends(oauth2_scheme)
) -> Union[str, User]:
    """Get current user from either API key or JWT token."""
    # Try API key first
    if api_key and api_key in settings.security.API_KEYS:
        return api_key
    
    # Then try JWT token
    if token:
        try:
            return await get_current_user_from_token(token)
        except HTTPException:
            pass
    
    # If neither works, raise an exception
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
