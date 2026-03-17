# API Documentation

## Overview

This document provides comprehensive documentation for the Emotion Recognition API with RAG (Retrieval-Augmented Generation) capabilities. The API implements a dual-head emotion recognition system with personalized learning, audio processing, and intelligent query capabilities.

**Base URL:** `http://localhost:8000` (development)

**API Version:** 1.0.0

---

## Table of Contents

1. [Authentication](#authentication)
2. [Audio Processing](#audio-processing)
3. [Feedback & Personalization](#feedback--personalization)
4. [Dashboard & Analytics](#dashboard--analytics)
5. [RAG (Query System)](#rag-query-system)
6. [Admin Endpoints](#admin-endpoints)
7. [System Endpoints](#system-endpoints)
8. [Error Handling](#error-handling)
9. [Architecture Overview](#architecture-overview)

---

## Authentication

All endpoints except `/auth/register`, `/auth/login`, `/`, and `/health` require JWT authentication.

**Authentication Header:**
```
Authorization: Bearer <jwt_token>
```

### POST `/api/auth/register`

Register a new user account.

**Request Body:**
```json
{
  "email": "user@example.com",
  "password": "securePassword123"
}
```

**Response (201 Created):**
```json
{
  "message": "User registered successfully",
  "user_id": "507f1f77bcf86cd799439011"
}
```

**Errors:**
- `400 Bad Request`: Invalid email format or password too weak
- `400 Bad Request`: Email already registered
- `500 Internal Server Error`: Registration failed

---

### POST `/api/auth/login`

Authenticate and receive JWT token.

**Request Body:**
```json
{
  "email": "user@example.com",
  "password": "securePassword123"
}
```

**Response (200 OK):**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "user_id": "507f1f77bcf86cd799439011"
}
```

**Errors:**
- `401 Unauthorized`: Incorrect email or password
- `500 Internal Server Error`: Login failed

**Token Usage:**
Store the `access_token` and include it in the `Authorization` header for all subsequent requests:
```
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

---

## Audio Processing

### POST `/api/audio/upload`

Upload and process audio file for emotion recognition.

**Authentication:** Required

**Request:**
- **Content-Type:** `multipart/form-data`
- **Body:**
  - `file`: Audio file (WAV, MP3, M4A, FLAC, OGG)
  - Max size: Configured in settings (typically 10MB)

**Processing Pipeline:**
1. Audio preprocessing (16kHz resampling, VAD, normalization)
2. Wav2Vec2 embedding extraction (768-dim)
3. Dual-head emotion classification (global + user heads)
4. Whisper transcription
5. Storage in MongoDB and Qdrant vector store

**Response (200 OK):**
```json
{
  "session_id": "507f1f77bcf86cd799439011",
  "emotion": "happy",
  "confidence": 0.78,
  "global_emotion": "happy",
  "global_confidence": 0.72,
  "user_emotion": "happy",
  "user_confidence": 0.85,
  "blend_weight": 0.55,
  "alpha_data": 0.67,
  "alpha_conf": 0.82,
  "alpha_formula": "sigmoid",
  "transcription": "I'm feeling great today!",
  "timestamp": "2026-02-14T10:30:00Z"
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | string | Unique identifier for this audio session |
| `emotion` | string | Final blended emotion prediction |
| `confidence` | float | Final confidence score (0-1) |
| `global_emotion` | string | Prediction from global head (shared model) |
| `global_confidence` | float | Confidence from global head |
| `user_emotion` | string\|null | Prediction from user head (personalized model) |
| `user_confidence` | float\|null | Confidence from user head |
| `blend_weight` | float | Alpha weight used for blending (0-1) |
| `alpha_data` | float\|null | Data component of alpha (feedback-based) |
| `alpha_conf` | float\|null | Confidence component of alpha |
| `alpha_formula` | string | Formula used: "sigmoid" or "linear" |
| `transcription` | string | Speech-to-text transcription |
| `timestamp` | datetime | Processing timestamp (ISO 8601) |

**Emotion Labels:**
- `neutral`
- `calm`
- `happy`
- `sad`
- `angry`
- `fearful`
- `disgusted`
- `surprised`

**Errors:**
- `400 Bad Request`: Invalid audio format or file too large
- `401 Unauthorized`: Missing or invalid JWT token
- `500 Internal Server Error`: Processing failed

**Performance:**
- Target latency: <5 seconds for 20-second audio
- Embedding extraction: ~500ms
- Classification: ~5ms
- Transcription: ~2-3 seconds

**Architecture Notes:**

**Phase 1 (Wav2Vec2 Migration):**
- Uses Wav2Vec2 neural embeddings instead of manual features
- 768-dimensional learned representations
- 5x faster than legacy feature extraction

**Phase 2 (Global Head):**
- Shared global classifier trained on public datasets
- Works for all users immediately
- Provides robust baseline predictions

**Phase 3 (Personalized Heads):**
- Per-user classifiers trained on feedback
- Adaptive blending based on feedback count and confidence
- Gradual transition from global to personalized predictions

**Phase 4 (Alpha Engine Refinement):**
- Sigmoid-based blending formula
- Separates data availability from confidence concerns
- Smooth transitions without hard clamping

---

## Feedback & Personalization

### POST `/api/audio/feedback`

Submit feedback to correct emotion predictions and train personalized models.

**Authentication:** Required

**Request Body:**
```json
{
  "session_id": "507f1f77bcf86cd799439011",
  "corrected_emotion": "sad"
}
```

**Response (200 OK):**
```json
{
  "status": "success",
  "feedback_count": 25,
  "training_triggered": true,
  "training_job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "message": "Feedback recorded (25 total). Your personalized model is being updated."
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Operation status ("success" or "error") |
| `feedback_count` | int | Total feedback samples provided by user |
| `training_triggered` | bool | Whether training was triggered |
| `training_job_id` | string\|null | Training job ID if triggered |
| `message` | string | Human-readable status message |

**Training Triggers:**
- **Initial training:** 20 feedback samples
- **Incremental training:** Every 10 samples after initial (30, 40, 50, etc.)

**Training Process:**
1. Feedback stored in MongoDB `user_feedback` collection
2. Background training job enqueued (non-blocking)
3. User head trained on all feedback data (20 epochs)
4. Model saved to MongoDB or filesystem
5. LRU cache invalidated for user

**Training Performance:**
- Training time: <10 seconds for 20-100 samples (CPU)
- Model size: ~5KB per user (compressed)
- Asynchronous execution (doesn't block API response)

**Errors:**
- `400 Bad Request`: Invalid emotion label or session_id
- `401 Unauthorized`: Missing or invalid JWT token
- `403 Forbidden`: Session belongs to another user
- `500 Internal Server Error`: Feedback submission failed

**Personalization Benefits:**
- Adapts to individual expression patterns
- Handles unique speaking styles (loud, flat pitch, sarcasm)
- Improves accuracy by 10%+ after 50 samples
- Maintains global baseline for new users

---

### GET `/api/training-status/{user_id}`

Query the status of the latest training job for a user.

**Authentication:** Required

**Path Parameters:**
- `user_id`: User identifier

**Authorization:**
- Users can only query their own training status
- Admins can query any user's status

**Response (200 OK):**
```json
{
  "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "completed",
  "created_at": "2026-02-14T10:30:00Z",
  "started_at": "2026-02-14T10:30:05Z",
  "completed_at": "2026-02-14T10:32:15Z",
  "error_message": null,
  "metrics": {
    "final_loss": 0.234,
    "final_accuracy": 0.89,
    "num_samples": 25,
    "num_epochs": 20,
    "storage_mode": "mongodb"
  }
}
```

**Job Status Values:**
- `queued`: Job is waiting to start
- `running`: Job is currently executing
- `completed`: Job finished successfully
- `failed`: Job encountered an error

**Response (200 OK - No Jobs):**
```json
{
  "job_id": null,
  "status": null,
  "created_at": null,
  "started_at": null,
  "completed_at": null,
  "error_message": null,
  "metrics": null
}
```

**Errors:**
- `401 Unauthorized`: Missing or invalid JWT token
- `403 Forbidden`: Cannot query other users' status
- `500 Internal Server Error`: Query failed

---

### POST `/api/trigger-training/{user_id}` (Admin Only)

Manually trigger training for a user, bypassing feedback count requirements.

**Authentication:** Required (Admin)

**Path Parameters:**
- `user_id`: User identifier

**Response (200 OK):**
```json
{
  "status": "success",
  "training_job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "message": "Training job enqueued for user 507f1f77bcf86cd799439011"
}
```

**Use Cases:**
- Testing training pipeline
- Retraining after model updates
- Recovering from failed training jobs
- Manual intervention for specific users

**Errors:**
- `400 Bad Request`: Invalid user_id format
- `401 Unauthorized`: Not authenticated as admin
- `404 Not Found`: User doesn't exist
- `500 Internal Server Error`: Training trigger failed

---

## Dashboard & Analytics

### GET `/api/dashboard/emotions/{user_id}`

Retrieve emotion analysis history for a user.

**Authentication:** Required

**Path Parameters:**
- `user_id`: User identifier

**Query Parameters:**
- `start_date` (optional): ISO 8601 datetime (e.g., "2026-02-01T00:00:00")
- `end_date` (optional): ISO 8601 datetime
- `limit` (optional): Max records to return (default: 100, max: 1000)

**Authorization:**
- Users can only access their own emotion data

**Response (200 OK):**
```json
{
  "emotions": [
    {
      "session_id": "507f1f77bcf86cd799439011",
      "emotion_label": "happy",
      "confidence": 0.78,
      "timestamp": "2026-02-14T10:30:00Z"
    },
    {
      "session_id": "507f1f77bcf86cd799439012",
      "emotion_label": "calm",
      "confidence": 0.82,
      "timestamp": "2026-02-14T09:15:00Z"
    }
  ],
  "total": 2
}
```

**Errors:**
- `400 Bad Request`: Invalid date format
- `401 Unauthorized`: Missing or invalid JWT token
- `403 Forbidden`: Cannot access other users' data
- `500 Internal Server Error`: Query failed

**Use Cases:**
- Emotion tracking over time
- Mood journaling
- Pattern analysis
- Historical review

---

### GET `/api/dashboard/stats/{user_id}`

Get aggregated statistics for a user.

**Authentication:** Required

**Path Parameters:**
- `user_id`: User identifier

**Authorization:**
- Users can only access their own statistics

**Response (200 OK):**
```json
{
  "total_sessions": 150,
  "emotion_distribution": {
    "happy": 45,
    "calm": 38,
    "neutral": 25,
    "sad": 18,
    "angry": 12,
    "surprised": 7,
    "fearful": 3,
    "disgusted": 2
  },
  "avg_confidence": 0.782
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `total_sessions` | int | Total audio sessions processed |
| `emotion_distribution` | object | Count of each emotion detected |
| `avg_confidence` | float | Average confidence across all predictions |

**Errors:**
- `401 Unauthorized`: Missing or invalid JWT token
- `403 Forbidden`: Cannot access other users' data
- `500 Internal Server Error`: Query failed

**Use Cases:**
- Overall mood trends
- Confidence tracking
- Usage statistics
- Data visualization

---

## RAG (Query System)

### POST `/api/rag/query`

Query the RAG system with natural language questions about your audio transcriptions.

**Authentication:** Required

**Request Body:**
```json
{
  "query": "What did I say about the project deadline?",
  "top_k": 5
}
```

**Request Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `query` | string | Natural language question |
| `top_k` | int | Number of relevant documents to retrieve (default: 5) |

**Response (200 OK):**
```json
{
  "answer": "Based on your recordings, you mentioned that the project deadline is next Friday, February 21st. You expressed concern about completing the design phase on time.",
  "sources": [
    {
      "session_id": "507f1f77bcf86cd799439011",
      "transcription": "The project deadline is next Friday...",
      "timestamp": "2026-02-14T10:30:00Z",
      "relevance_score": 0.92
    }
  ],
  "query": "What did I say about the project deadline?"
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `answer` | string | AI-generated answer based on transcriptions |
| `sources` | array | Relevant audio sessions used for answer |
| `query` | string | Original query (echoed back) |

**Source Object:**

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | string | Audio session identifier |
| `transcription` | string | Relevant transcription excerpt |
| `timestamp` | datetime | When audio was recorded |
| `relevance_score` | float | Similarity score (0-1) |

**Special Responses:**

**No Data Available:**
```json
{
  "answer": "I don't have any audio transcriptions to search through yet. Please upload some audio files first, and then I'll be able to answer your questions based on your recordings.",
  "sources": [],
  "query": "What did I say about the project deadline?"
}
```

**Errors:**
- `401 Unauthorized`: Missing or invalid JWT token
- `503 Service Unavailable`: AI service (Ollama) unavailable
- `500 Internal Server Error`: Query processing failed

**Architecture:**
1. Query embedding generated using sentence transformers
2. Vector similarity search in Qdrant (user-specific collection)
3. Top-K relevant transcriptions retrieved
4. Context passed to Ollama LLM for answer generation
5. Answer returned with source citations

**Use Cases:**
- Search through audio recordings
- Find specific information mentioned
- Summarize topics discussed
- Memory augmentation

---

## Admin Endpoints

### GET `/admin/user-models`

List all user models with metadata.

**Authentication:** Required (Admin)

**Response (200 OK):**
```json
{
  "total_count": 150,
  "models": [
    {
      "user_id": "507f1f77bcf86cd799439011",
      "feedback_count": 50,
      "model_size_kb": 4.8,
      "last_trained": "2026-02-14T10:30:00Z"
    },
    {
      "user_id": "507f1f77bcf86cd799439012",
      "feedback_count": 35,
      "model_size_kb": 4.9,
      "last_trained": "2026-02-13T15:20:00Z"
    }
  ]
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `total_count` | int | Total number of user models |
| `models` | array | List of user model metadata |

**Model Metadata:**

| Field | Type | Description |
|-------|------|-------------|
| `user_id` | string | User identifier |
| `feedback_count` | int | Number of feedback samples |
| `model_size_kb` | float | Model file size in KB |
| `last_trained` | datetime | Last training timestamp |

**Errors:**
- `401 Unauthorized`: Not authenticated as admin
- `500 Internal Server Error`: Query failed

**Use Cases:**
- Monitor personalization adoption
- Track storage usage
- Identify inactive models
- System health monitoring

---

### DELETE `/admin/user-models/cleanup`

Delete inactive user models based on criteria.

**Authentication:** Required (Admin)

**Request Body:**
```json
{
  "min_days_inactive": 90,
  "max_feedback_count": 10,
  "user_ids": ["507f1f77bcf86cd799439011"]
}
```

**Request Fields (all optional):**

| Field | Type | Description |
|-------|------|-------------|
| `min_days_inactive` | int | Minimum days since last training |
| `max_feedback_count` | int | Maximum feedback count to consider inactive |
| `user_ids` | array | Specific user IDs to delete |

**Response (200 OK):**
```json
{
  "deleted_count": 15,
  "deleted_user_ids": [
    "507f1f77bcf86cd799439011",
    "507f1f77bcf86cd799439012"
  ],
  "total_size_freed_kb": 72.5
}
```

**Errors:**
- `401 Unauthorized`: Not authenticated as admin
- `500 Internal Server Error`: Cleanup failed

**Use Cases:**
- Free up storage space
- Remove abandoned models
- Maintenance operations
- Cost optimization

---

## System Endpoints

### GET `/`

Root endpoint providing API information.

**Authentication:** Not required

**Response (200 OK):**
```json
{
  "message": "RAG Backend with Audio Emotion Analysis API",
  "version": "1.0.0",
  "docs": "/docs"
}
```

---

### GET `/health`

Health check endpoint for monitoring.

**Authentication:** Not required

**Response (200 OK):**
```json
{
  "status": "healthy"
}
```

**Use Cases:**
- Load balancer health checks
- Monitoring systems
- Uptime tracking
- Deployment verification

---

## Error Handling

### Standard Error Response

All errors follow this format:

```json
{
  "detail": "Error message describing what went wrong"
}
```

### HTTP Status Codes

| Code | Meaning | Common Causes |
|------|---------|---------------|
| 200 | OK | Request succeeded |
| 201 | Created | Resource created successfully |
| 400 | Bad Request | Invalid input, validation failed |
| 401 | Unauthorized | Missing or invalid JWT token |
| 403 | Forbidden | Insufficient permissions |
| 404 | Not Found | Resource doesn't exist |
| 500 | Internal Server Error | Server-side error |
| 503 | Service Unavailable | External service unavailable |

### Common Error Scenarios

**Authentication Errors:**
```json
{
  "detail": "Could not validate credentials"
}
```

**Validation Errors:**
```json
{
  "detail": "Invalid emotion label: 'excited'. Must be one of: neutral, calm, happy, sad, angry, fearful, disgusted, surprised"
}
```

**Permission Errors:**
```json
{
  "detail": "You can only query your own training status"
}
```

**Service Unavailable:**
```json
{
  "detail": "The AI service is temporarily unavailable. Please try again later."
}
```

---

## Architecture Overview

### System Components

```
┌─────────────────────────────────────────────────────┐
│              Audio Processing Pipeline              │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Audio Upload → Preprocessing → Wav2Vec2 Encoder   │
│                                                     │
│  ┌──────────────┐         ┌──────────────┐        │
│  │ Global Head  │         │  User Head   │        │
│  │  (shared)    │         │  (per-user)  │        │
│  └──────┬───────┘         └──────┬───────┘        │
│         │                        │                 │
│         ▼                        ▼                 │
│       P_g                      P_u                 │
│         │                        │                 │
│         └────────┬───────────────┘                 │
│                  │                                  │
│                  ▼                                  │
│           Alpha Engine                             │
│      (Sigmoid-Based Blending)                      │
│                  │                                  │
│                  ▼                                  │
│          Final Prediction                          │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### Data Flow

1. **Audio Upload** → User uploads audio file
2. **Preprocessing** → 16kHz resampling, VAD, normalization
3. **Embedding** → Wav2Vec2 extracts 768-dim neural representation
4. **Global Prediction** → Shared model predicts emotion (P_g)
5. **User Prediction** → Personalized model predicts emotion (P_u) if available
6. **Alpha Computation** → Sigmoid formula computes blending weight
7. **Blending** → P_f = α·P_g + (1-α)·P_u
8. **Transcription** → Whisper converts speech to text
9. **Storage** → MongoDB (metadata) + Qdrant (vectors)
10. **Response** → Return emotion, confidence, transcription

### Storage Architecture

**MongoDB Collections:**
- `users`: User accounts and authentication
- `audio_sessions`: Audio session metadata
- `emotion_analysis`: Emotion predictions and embeddings
- `user_feedback`: Feedback corrections for training
- `training_jobs`: Training job status tracking
- `user_models`: User head model blobs (Phase 5)

**Qdrant Collections:**
- `user_{user_id}`: Per-user vector store for transcriptions

**Filesystem (Phase 2-4):**
- `models/global_emotion_head.pt`: Shared global classifier
- `models/user_heads/{user_id}.pt`: Per-user classifiers

**MongoDB Blob Storage (Phase 5):**
- User heads stored as compressed BSON Binary
- Fallback to filesystem if MongoDB unavailable

### Alpha Engine (Blending Strategy)

**Sigmoid Formula:**
```
alpha_data = 1 / (1 + N/K)
alpha_conf = sigmoid(β(C_g - τ))
alpha = alpha_data × alpha_conf
```

**Parameters:**
- K = 50: Feedback scaling constant
- τ = 0.6: Confidence threshold
- β = 10: Sigmoid sharpness

**Behavior:**
- New users (N=0): Trust global head, modulated by confidence
- Medium feedback (N=50): Balanced blending
- High feedback (N=100+): Favor user head unless global very confident

### Performance Characteristics

| Operation | Target Latency | Notes |
|-----------|----------------|-------|
| Audio upload | <5s | For 20-second audio |
| Embedding extraction | ~500ms | Wav2Vec2 inference |
| Classification | ~5ms | Dual-head forward pass |
| Transcription | ~2-3s | Whisper inference |
| Feedback submission | <100ms | Async training |
| User head training | <10s | 20-100 samples, CPU |
| RAG query | <2s | Vector search + LLM |

### Scalability

- **Users:** 10,000+ supported
- **User models:** ~5KB each (50MB for 10K users)
- **LRU cache:** 100 models in memory (~500KB)
- **Concurrent requests:** 1000+ with proper infrastructure
- **Storage:** MongoDB for centralized multi-server deployment

---

## Migration Phases

### Phase 1: Wav2Vec2 Migration (Completed)
- Replaced manual features with neural embeddings
- 768-dimensional Wav2Vec2 representations
- 5x faster processing
- More robust to noise and silence

### Phase 2: Global Head (Completed)
- Shared emotion classifier for all users
- Trained on RAVDESS + CREMA-D datasets
- Immediate accuracy for new users
- Baseline for personalization

### Phase 3: Personalized Heads (Completed)
- Per-user classifiers trained on feedback
- Adaptive blending strategy
- 10%+ accuracy improvement after 50 samples
- Graceful transition from global to personalized

### Phase 4: Alpha Engine Refinement (Completed)
- Sigmoid-based blending formula
- Separates data availability from confidence
- Smooth transitions without hard clamping
- Tunable hyperparameters (K, τ, β)

### Phase 5: MongoDB Storage (In Progress)
- Migrate user heads from filesystem to MongoDB
- Centralized storage for multi-server deployment
- Compressed BSON Binary storage
- Backward compatible with file fallback

---

## API Client Examples

### Python Example

```python
import requests

# Base URL
BASE_URL = "http://localhost:8000"

# 1. Register
response = requests.post(f"{BASE_URL}/api/auth/register", json={
    "email": "user@example.com",
    "password": "securePassword123"
})
user_id = response.json()["user_id"]

# 2. Login
response = requests.post(f"{BASE_URL}/api/auth/login", json={
    "email": "user@example.com",
    "password": "securePassword123"
})
token = response.json()["access_token"]

# 3. Upload audio
headers = {"Authorization": f"Bearer {token}"}
files = {"file": open("audio.wav", "rb")}
response = requests.post(f"{BASE_URL}/api/audio/upload", headers=headers, files=files)
result = response.json()
print(f"Emotion: {result['emotion']}, Confidence: {result['confidence']}")

# 4. Submit feedback
response = requests.post(f"{BASE_URL}/api/audio/feedback", headers=headers, json={
    "session_id": result["session_id"],
    "corrected_emotion": "happy"
})
print(f"Feedback count: {response.json()['feedback_count']}")

# 5. Query RAG
response = requests.post(f"{BASE_URL}/api/rag/query", headers=headers, json={
    "query": "What did I say about the project?",
    "top_k": 5
})
print(f"Answer: {response.json()['answer']}")
```

### JavaScript Example

```javascript
const BASE_URL = "http://localhost:8000";

// 1. Register
const registerResponse = await fetch(`${BASE_URL}/api/auth/register`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    email: "user@example.com",
    password: "securePassword123"
  })
});
const { user_id } = await registerResponse.json();

// 2. Login
const loginResponse = await fetch(`${BASE_URL}/api/auth/login`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    email: "user@example.com",
    password: "securePassword123"
  })
});
const { access_token } = await loginResponse.json();

// 3. Upload audio
const formData = new FormData();
formData.append("file", audioFile);

const uploadResponse = await fetch(`${BASE_URL}/api/audio/upload`, {
  method: "POST",
  headers: { "Authorization": `Bearer ${access_token}` },
  body: formData
});
const result = await uploadResponse.json();
console.log(`Emotion: ${result.emotion}, Confidence: ${result.confidence}`);

// 4. Submit feedback
const feedbackResponse = await fetch(`${BASE_URL}/api/audio/feedback`, {
  method: "POST",
  headers: {
    "Authorization": `Bearer ${access_token}`,
    "Content-Type": "application/json"
  },
  body: JSON.stringify({
    session_id: result.session_id,
    corrected_emotion: "happy"
  })
});
const feedbackResult = await feedbackResponse.json();
console.log(`Feedback count: ${feedbackResult.feedback_count}`);

// 5. Query RAG
const ragResponse = await fetch(`${BASE_URL}/api/rag/query`, {
  method: "POST",
  headers: {
    "Authorization": `Bearer ${access_token}`,
    "Content-Type": "application/json"
  },
  body: JSON.stringify({
    query: "What did I say about the project?",
    top_k: 5
  })
});
const ragResult = await ragResponse.json();
console.log(`Answer: ${ragResult.answer}`);
```

---

## Interactive API Documentation

FastAPI provides interactive API documentation at:

- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

These interfaces allow you to:
- Explore all endpoints
- View request/response schemas
- Test endpoints directly in the browser
- See example requests and responses
- Understand authentication requirements

---

## Support & Contact

For issues, questions, or feature requests, please contact the development team or file an issue in the project repository.

**Version:** 1.0.0  
**Last Updated:** February 14, 2026
