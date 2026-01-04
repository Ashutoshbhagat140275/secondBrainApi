from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from bson import ObjectId
from app.config import settings
from app.db.mongodb import get_database
from app.models.user import User

security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> dict:
    """
    Verify JWT token and return user information
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        token = credentials.credentials
        payload = jwt.decode(
            token, 
            settings.jwt_secret_key, 
            algorithms=[settings.jwt_algorithm]
        )
        user_id: str = payload.get("user_id")
        email: str = payload.get("email")
        
        if user_id is None or email is None:
            raise credentials_exception
        
        # Verify user exists in database
        db = get_database()
        user_collection = User.get_collection(db)
        try:
            user = user_collection.find_one({"_id": ObjectId(user_id)})
        except Exception:
            raise credentials_exception
        
        if user is None:
            raise credentials_exception
        
        return {"user_id": user_id, "email": email}
    
    except JWTError:
        raise credentials_exception


async def get_current_user_id(
    current_user: dict = Depends(get_current_user)
) -> str:
    """Extract user_id from current user"""
    return current_user["user_id"]

