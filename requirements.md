# Requirements Document - RAG Backend with Audio Emotion Analysis

## 1. Project Overview

This project implements a multi-tenant RAG (Retrieval-Augmented Generation) backend system that processes audio files, analyzes emotions using MFCC features, converts speech to text, and provides intelligent querying capabilities through vector search and local LLM integration.

## 2. Functional Requirements

### 2.1 Authentication & Authorization

**FR-1.1: User Registration**
- System shall allow users to register with email and password
- Passwords must be securely hashed before storage
- Email validation required
- Duplicate email registration prevention

**FR-1.2: User Login**
- System shall authenticate users with email and password
- Upon successful authentication, system shall generate JWT token
- JWT token shall include user_id in payload
- Token expiration time: 24 hours (configurable)

**FR-1.3: JWT Token Validation**
- All protected endpoints shall validate JWT token
- Invalid or expired tokens shall be rejected
- User identity extracted from token for multi-tenant isolation

### 2.2 Audio Processing Pipeline

**FR-2.1: Audio Upload**
- System shall accept audio file uploads via POST API
- Supported formats: WAV, MP3, M4A, FLAC
- Maximum file size: 50MB (configurable)
- Audio file validation required

**FR-2.2: MFCC Feature Extraction**
- System shall extract Mel-frequency cepstral coefficients (MFCC) from audio
- MFCC parameters: 13 coefficients, 0.025s frame length, 0.01s hop length
- Feature extraction using librosa library

**FR-2.3: Emotion Analysis**
- System shall classify emotions from MFCC features
- Emotion categories: happy, sad, angry, neutral, fearful, surprised, disgusted
- Confidence scores for each emotion prediction
- Store raw MFCC features and predictions in database

**FR-2.4: Speech-to-Text Conversion**
- System shall transcribe audio to text using Whisper model
- Support for multiple languages (auto-detect)
- Store transcription text in vector database and metadata store

**FR-2.5: Vector Storage**
- System shall generate embeddings for transcribed text
- Store embeddings in Qdrant vector database
- User-specific collection isolation: `user_{user_id}_documents`
- Metadata storage: session_id, timestamp, text content

### 2.3 RAG Query System

**FR-3.1: Query Processing**
- System shall accept natural language queries
- Generate query embeddings using same model as document embeddings
- Search user-specific vector collection for relevant documents
- Retrieve top-k most similar documents (k=5, configurable)

**FR-3.2: Context-Aware Response Generation**
- System shall build context prompt with retrieved documents
- Query local LLM (Ollama) with context and user query
- Return generated response to user
- Maintain conversation context if needed

**FR-3.3: Multi-Tenant Isolation**
- All queries scoped to user's own vector collection
- User cannot access other users' data
- Enforced at service layer and database level

### 2.4 Dashboard & Analytics

**FR-4.1: Emotion Analysis Dashboard**
- System shall provide API endpoint to retrieve emotion analysis data
- Filter by user_id (from authenticated token)
- Return emotion labels, confidence scores, timestamps
- Support date range filtering

**FR-4.2: User Statistics**
- System shall provide aggregated statistics per user
- Metrics: total audio sessions, emotion distribution, average confidence
- Time-based analytics (daily, weekly, monthly)

## 3. Non-Functional Requirements

### 3.1 Performance

**NFR-1.1: Response Time**
- Audio processing: < 30 seconds for 1-minute audio file
- RAG query response: < 5 seconds
- Dashboard data retrieval: < 2 seconds

**NFR-1.2: Scalability**
- Support concurrent audio processing (queue-based)
- Vector search optimized for up to 10,000 documents per user
- Database connection pooling

### 3.2 Security

**NFR-2.1: Data Protection**
- JWT secrets stored in environment variables
- Password hashing using bcrypt (12 rounds)
- HTTPS required in production
- Input validation and sanitization

**NFR-2.2: Multi-Tenancy Security**
- Strict user data isolation
- No cross-user data access
- User_id validation on all operations

### 3.3 Reliability

**NFR-3.1: Error Handling**
- Graceful error handling for all API endpoints
- Meaningful error messages
- Logging for debugging and monitoring

**NFR-3.2: Data Persistence**
- MongoDB for metadata (durable storage)
- Qdrant for vectors (persistent collections)
- Backup and recovery procedures

### 3.4 Maintainability

**NFR-4.1: Code Quality**
- Modular architecture
- Clear separation of concerns
- Comprehensive documentation
- Type hints and validation

## 4. Technical Constraints

**TC-1:** Python 3.10+ required
**TC-2:** FastAPI framework for REST API
**TC-3:** Qdrant vector database (self-hosted)
**TC-4:** MongoDB for user data storage
**TC-5:** Local LLM via Ollama (no external API dependencies)
**TC-6:** MFCC-based emotion analysis (custom model)

## 5. API Endpoints Specification

### 5.1 Authentication Endpoints

**POST /api/auth/register**
- Request: `{email: string, password: string}`
- Response: `{message: string, user_id: string}`

**POST /api/auth/login**
- Request: `{email: string, password: string}`
- Response: `{access_token: string, token_type: string, user_id: string}`

### 5.2 Audio Processing Endpoint

**POST /api/audio/upload**
- Headers: `Authorization: Bearer {token}`
- Request: Multipart form data with audio file
- Response: `{session_id: string, emotion: string, confidence: float, transcription: string, timestamp: string}`

### 5.3 RAG Query Endpoint

**POST /api/rag/query**
- Headers: `Authorization: Bearer {token}`
- Request: `{query: string, top_k: int (optional, default=5)}`
- Response: `{answer: string, sources: array, query: string}`

### 5.4 Dashboard Endpoints

**GET /api/dashboard/emotions/{user_id}**
- Headers: `Authorization: Bearer {token}`
- Query params: `start_date?`, `end_date?`, `limit?`
- Response: `{emotions: array, total: int}`

**GET /api/dashboard/stats/{user_id}**
- Headers: `Authorization: Bearer {token}`
- Response: `{total_sessions: int, emotion_distribution: object, avg_confidence: float}`

## 6. Data Models

### 6.1 User Model
```python
{
    "_id": ObjectId,
    "email": string (unique),
    "password_hash": string,
    "created_at": datetime
}
```

### 6.2 Audio Session Model
```python
{
    "_id": ObjectId,
    "user_id": string,
    "audio_file_path": string,
    "timestamp": datetime,
    "emotion_data": {
        "label": string,
        "confidence": float
    },
    "transcription_text": string,
    "qdrant_collection_id": string
}
```

### 6.3 Emotion Analysis Model
```python
{
    "_id": ObjectId,
    "user_id": string,
    "session_id": string,
    "emotion_label": string,
    "confidence": float,
    "mfcc_features": array,
    "timestamp": datetime
}
```

## 7. Dependencies

- FastAPI 0.104+
- Uvicorn (ASGI server)
- python-jose[cryptography] (JWT)
- passlib[bcrypt] (password hashing)
- pymongo (MongoDB client)
- qdrant-client (Qdrant client)
- librosa (audio processing, MFCC)
- soundfile (audio I/O)
- openai-whisper / faster-whisper (speech-to-text)
- sentence-transformers (embeddings)
- ollama (local LLM client)
- scikit-learn / tensorflow (emotion model)
- python-multipart (file uploads)
- pydantic (data validation)

## 8. Environment Variables

- `MONGODB_URL`: MongoDB connection string
- `QDRANT_URL`: Qdrant server URL
- `QDRANT_API_KEY`: Qdrant API key (if required)
- `JWT_SECRET_KEY`: Secret for JWT token signing
- `JWT_ALGORITHM`: JWT algorithm (HS256)
- `JWT_EXPIRATION_HOURS`: Token expiration time
- `OLLAMA_BASE_URL`: Ollama server URL (default: http://localhost:11434)
- `OLLAMA_MODEL`: Model name for RAG (e.g., llama2, mistral)
- `EMBEDDING_MODEL`: Model for generating embeddings
- `AUDIO_UPLOAD_DIR`: Directory for temporary audio storage
- `MAX_AUDIO_SIZE_MB`: Maximum audio file size in MB

## 9. Future Enhancements (Out of Scope)

- Real-time audio streaming
- WebSocket support for live transcription
- Advanced emotion visualization
- Multi-language emotion models
- Audio quality assessment
- Batch audio processing
- Admin dashboard
- User role management

