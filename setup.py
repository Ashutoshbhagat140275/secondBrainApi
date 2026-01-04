"""
Setup script to initialize the project
"""
import os
from pathlib import Path
from app.config import settings


def create_directories():
    """Create necessary directories"""
    directories = [
        settings.audio_upload_dir,
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")


def check_env_file():
    """Check if .env file exists"""
    if not os.path.exists(".env"):
        print("Warning: .env file not found. Please create one from .env.example")
        return False
    return True


if __name__ == "__main__":
    print("Setting up project...")
    create_directories()
    
    if check_env_file():
        print("Setup complete!")
    else:
        print("Setup incomplete. Please create .env file.")

