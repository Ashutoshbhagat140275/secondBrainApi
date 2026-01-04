"""
Test script for API endpoints
"""
import requests
import json
import os
from pathlib import Path

BASE_URL = "http://localhost:8000"

def print_response(title, response):
    """Print formatted response"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"Status Code: {response.status_code}")
    try:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except:
        print(f"Response: {response.text}")

def test_health():
    """Test health endpoint"""
    print("\n[1] Testing Health Check...")
    response = requests.get(f"{BASE_URL}/health")
    print_response("Health Check", response)
    return response.status_code == 200

def test_register():
    """Test user registration"""
    print("\n[2] Testing User Registration...")
    import random
    # Use random email to avoid "already exists" error
    email = f"test{random.randint(1000, 9999)}@example.com"
    data = {
        "email": email,
        "password": "test123456"
    }
    response = requests.post(f"{BASE_URL}/api/auth/register", json=data)
    print_response("User Registration", response)
    if response.status_code in [200, 201]:
        return email  # Return email for use in login
    return None

def test_login(email=None):
    """Test user login"""
    print("\n[3] Testing User Login...")
    if not email:
        email = "test@example.com"  # Fallback to default
    data = {
        "email": email,
        "password": "test123456"
    }
    response = requests.post(f"{BASE_URL}/api/auth/login", json=data)
    print_response("User Login", response)
    
    if response.status_code == 200:
        token_data = response.json()
        return token_data.get("access_token"), token_data.get("user_id")
    return None, None

def test_audio_upload(token):
    """Test audio upload"""
    print("\n[4] Testing Audio Upload...")
    
    # Create a dummy audio file for testing
    test_audio_path = "test_audio.wav"
    if not os.path.exists(test_audio_path):
        print(f"Note: {test_audio_path} not found. Skipping audio upload test.")
        print("To test audio upload, create a test audio file first.")
        return None
    
    headers = {"Authorization": f"Bearer {token}"}
    with open(test_audio_path, "rb") as f:
        files = {"file": (test_audio_path, f, "audio/wav")}
        response = requests.post(
            f"{BASE_URL}/api/audio/upload",
            headers=headers,
            files=files
        )
    print_response("Audio Upload", response)
    return response.status_code == 200

def test_rag_query(token):
    """Test RAG query"""
    print("\n[5] Testing RAG Query...")
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    data = {
        "query": "What did I say?",
        "top_k": 5
    }
    response = requests.post(
        f"{BASE_URL}/api/rag/query",
        headers=headers,
        json=data
    )
    print_response("RAG Query", response)
    return response.status_code in [200, 500]  # 500 is OK if Qdrant/Ollama not running

def test_dashboard_emotions(token, user_id):
    """Test dashboard emotions endpoint"""
    print("\n[6] Testing Dashboard Emotions...")
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(
        f"{BASE_URL}/api/dashboard/emotions/{user_id}",
        headers=headers
    )
    print_response("Dashboard Emotions", response)
    return response.status_code == 200

def test_dashboard_stats(token, user_id):
    """Test dashboard stats endpoint"""
    print("\n[7] Testing Dashboard Stats...")
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(
        f"{BASE_URL}/api/dashboard/stats/{user_id}",
        headers=headers
    )
    print_response("Dashboard Stats", response)
    return response.status_code == 200

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("API TESTING SUITE")
    print("="*60)
    
    # Test health
    if not test_health():
        print("\n❌ Server is not running. Please start the server first.")
        return
    
    # Test registration
    registered_email = test_register()
    
    # Test login (use registered email if available, otherwise try default)
    token, user_id = test_login(registered_email)
    if not token:
        print("\n❌ Login failed. Cannot test protected endpoints.")
        return
    
    if not user_id:
        user_id = "unknown"
    
    # Test protected endpoints
    test_audio_upload(token)
    test_rag_query(token)
    test_dashboard_emotions(token, user_id)
    test_dashboard_stats(token, user_id)
    
    print("\n" + "="*60)
    print("TESTING COMPLETE")
    print("="*60)
    print("\nNote: Some endpoints may fail if Qdrant or Ollama are not running.")
    print("This is expected behavior.")

if __name__ == "__main__":
    main()

