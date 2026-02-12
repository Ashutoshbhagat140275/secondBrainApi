from datetime import datetime, timedelta
from jose import jwt
from passlib.context import CryptContext
from app.config import settings
from app.db.mongodb import get_database
from app.models.user import User
from typing import Optional
import bcrypt

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(
        password.encode("utf-8"),
        hashed_password.encode("utf-8")
    )

def get_password_hash(password: str) -> str:
    hashed = bcrypt.hashpw(
        password.encode("utf-8"),
        bcrypt.gensalt()
    )
    return hashed.decode("utf-8")


def create_access_token(data: dict) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(hours=settings.jwt_expiration_hours)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode, 
        settings.jwt_secret_key, 
        algorithm=settings.jwt_algorithm
    )
    return encoded_jwt


async def authenticate_user(email: str, password: str) -> Optional[User]:
    """Authenticate user with email and password"""
    db = get_database()
    user_collection = User.get_collection(db)
    
    user_data = user_collection.find_one({"email": email})
    if not user_data:
        return None
    
    if not verify_password(password, user_data["password_hash"]):
        return None
    
    return User.from_dict(user_data)


async def create_user(email: str, password: str) -> User:
    """Create a new user"""
    db = get_database()
    print("before the get_collection")
    user_collection = User.get_collection(db)
    
    # Check if user already exists
    print("before the user call")
    existing_user = user_collection.find_one({"email": email})
    if existing_user:
        raise ValueError("User with this email already exists")
    
    # Create new user
    print("about to create the password")
    password_hash = get_password_hash(password)
    print("password created successful")
    user = User(email=email, password_hash=password_hash)
    
    # Insert into database
    result = user_collection.insert_one(user.to_dict())
    user._id = result.inserted_id
    
    return user

