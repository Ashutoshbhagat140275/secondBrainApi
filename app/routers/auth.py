from fastapi import APIRouter, HTTPException, status
from app.schemas.auth import UserRegister, UserLogin, TokenResponse, RegisterResponse
from app.services.auth import authenticate_user, create_user, create_access_token
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


@router.post("/register", response_model=RegisterResponse)
async def register(user_data: UserRegister):
    """Register a new user"""
    try:
        user = await create_user(user_data.email, user_data.password)
        return RegisterResponse(
            message="User registered successfully",
            user_id=str(user._id)
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )


@router.post("/login", response_model=TokenResponse)
async def login(user_data: UserLogin):
    """Login and get JWT token"""
    user = await authenticate_user(user_data.email, user_data.password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(
        data={"user_id": str(user._id), "email": user.email}
    )
    
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        user_id=str(user._id)
    )

