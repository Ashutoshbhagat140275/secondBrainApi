# Design Document - RAG Backend with Audio Emotion Analysis

## 1. System Architecture

### 1.1 High-Level Architecture

```
┌─────────────┐
│   Client    │
│  (Frontend) │
└──────┬──────┘
       │ HTTP/REST
       │
┌──────▼─────────────────────────────────────┐
│         FastAPI Backend                    │
│  ┌──────────────────────────────────────┐ │
│  │  Authentication Layer (JWT)          │ │
│  └──────────────────────────────────────┘ │
│  ┌──────────────────────────────────────┐ │
│  │  API Routers                         │ │
│  │  - /auth, /audio, /rag, /dashboard  │ │
│  └──────────────────────────────────────┘ │
│  ┌──────────────────────────────────────┐ │
│  │  Service Layer                       │ │
│  │  - Audio Processing                  │ │
│  │  - Emotion Analysis                  │ │
│  │  - Transcription                     │ │
│  │  - Vector Store                      │ │
│  │  - RAG Service                       │ │
│  └──────────────────────────────────────┘ │
└──────┬─────────────────────────────────────┘
       │
       ├─────────────────┬──────────────────┐
       │                 │                  │
┌──────▼──────┐  ┌──────▼──────┐  ┌───────▼──────┐
│   MongoDB   │  │   Qdrant    │  │    Ollama    │
│  (Metadata) │  │  (Vectors)  │  │  (LLM/RAG)   │
└─────────────┘  └─────────────┘  └──────────────┘
```

### 1.2 Component Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Application                   │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   Routers    │  │  Middleware  │  │   Services   │ │
│  │              │  │              │  │              │ │
│  │ - auth.py    │  │ - auth.py    │  │ - auth.py    │ │
│  │ - audio.py   │  │  (JWT)       │  │ - audio_     │ │
│  │ - rag.py     │  │              │  │   processor  │ │
│  │ - dashboard  │  │              │  │ - emotion_   │ │
│  │   .py        │  │              │  │   analyzer   │ │
│  └──────┬───────┘  └──────┬───────┘  │ - transcript │ │
│         │                 │          │   ion.py     │ │
│         └─────────┬───────┘          │ - vector_    │ │
│                   │                  │   store.py   │ │
│                   │                  │ - rag_       │ │
│                   │                  │   service.py │ │
│                   │                  └──────┬───────┘ │
│                   │                         │         │
│  ┌────────────────▼─────────────────────────▼───────┐ │
│  │              Database Layer                       │ │
│  │  ┌──────────────┐         ┌──────────────┐      │ │
│  │  │  mongodb.py  │         │  qdrant.py   │      │ │
│  │  └──────────────┘         └──────────────┘      │ │
│  └──────────────────────────────────────────────────┘ │
│                                                         │
│  ┌──────────────────────────────────────────────────┐ │
│  │              Models & Schemas                     │ │
│  │  - User, AudioSession, EmotionAnalysis           │ │
│  │  - Request/Response schemas (Pydantic)           │ │
│  └──────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## 2. Database Design

### 2.1 MongoDB Schema

#### Users Collection
```javascript
{
  "_id": ObjectId("..."),
  "email": "user@example.com",  // Indexed, unique
  "password_hash": "$2b$12$...",
  "created_at": ISODate("2024-01-01T00:00:00Z")
}
```

**Indexes:**
- `email`: unique index

#### Audio Sessions Collection
```javascript
{
  "_id": ObjectId("..."),
  "user_id": "user_123",
  "audio_file_path": "/uploads/user_123/session_456.wav",
  "timestamp": ISODate("2024-01-01T12:00:00Z"),
  "emotion_data": {
    "label": "happy",
    "confidence": 0.87
  },
  "transcription_text": "Hello, this is a test transcription.",
  "qdrant_collection_id": "user_user_123_documents"
}
```

**Indexes:**
- `user_id`: index
- `timestamp`: index
- Compound: `{user_id: 1, timestamp: -1}`

#### Emotion Analyses Collection
```javascript
{
  "_id": ObjectId("..."),
  "user_id": "user_123",
  "session_id": "session_456",
  "emotion_label": "happy",
  "confidence": 0.87,
  "mfcc_features": [0.12, -0.45, 0.78, ...],  // 13 coefficients per frame
  "timestamp": ISODate("2024-01-01T12:00:00Z")
}
```

**Indexes:**
- `user_id`: index
- `session_id`: index
- Compound: `{user_id: 1, timestamp: -1}`

### 2.2 Qdrant Collections

#### Collection Naming Convention
- Format: `user_{user_id}_documents`
- Example: `user_abc123_documents`

#### Document Structure
```json
{
  "id": "doc_12345",
  "vector": [0.1, 0.2, ..., 0.9],  // 384-dim embedding (sentence-transformers)
  "payload": {
    "text": "Hello, this is a test transcription.",
    "session_id": "session_456",
    "user_id": "user_123",
    "timestamp": "2024-01-01T12:00:00Z",
    "emotion_label": "happy"
  }
}
```

**Collection Configuration:**
- Distance metric: Cosine
- Vector size: 384 (or based on embedding model)
- On-disk persistence: enabled

## 3. Data Flow

### 3.1 Audio Processing Flow

```
1. Client uploads audio file
   ↓
2. FastAPI receives file (multipart/form-data)
   ↓
3. Validate file (format, size)
   ↓
4. Save to temporary storage
   ↓
5. Extract MFCC features (librosa)
   ↓
6. Classify emotion (custom model)
   ↓
7. Transcribe audio (Whisper)
   ↓
8. Generate embeddings (sentence-transformers)
   ↓
9. Store in Qdrant (user-specific collection)
   ↓
10. Store metadata in MongoDB
    ↓
11. Return response to client
```

### 3.2 RAG Query Flow

```
1. Client sends query with JWT token
   ↓
2. Validate token, extract user_id
   ↓
3. Generate query embedding
   ↓
4. Search Qdrant collection: user_{user_id}_documents
   ↓
5. Retrieve top-k documents (default k=5)
   ↓
6. Build context prompt:
   Context: [retrieved documents]
   Question: [user query]
   ↓
7. Query Ollama LLM with prompt
   ↓
8. Return generated answer + sources
```

### 3.3 Authentication Flow

```
Registration:
1. Client → POST /api/auth/register {email, password}
2. Hash password (bcrypt)
3. Store in MongoDB (users collection)
4. Return success

Login:
1. Client → POST /api/auth/login {email, password}
2. Verify password
3. Generate JWT token (payload: {user_id, email})
4. Return token

Protected Endpoints:
1. Client → Request with Authorization header
2. Extract and validate JWT token
3. Extract user_id from token
4. Process request with user context
```

## 4. Multi-Tenancy Design

### 4.1 Isolation Strategy

**Data Isolation:**
- Qdrant: Separate collection per user (`user_{user_id}_documents`)
- MongoDB: All queries filtered by `user_id`
- File storage: User-specific directories

**Access Control:**
- JWT token contains `user_id`
- All service methods require `user_id` parameter
- Database queries always include `user_id` filter
- No cross-user data access possible

### 4.2 Implementation Pattern

```python
# Service layer pattern
async def process_audio(user_id: str, audio_file: UploadFile):
    # All operations scoped to user_id
    collection_name = f"user_{user_id}_documents"
    # ... process and store
    await mongo_db.audio_sessions.insert_one({
        "user_id": user_id,
        ...
    })
```

## 5. Emotion Analysis Model Design

### 5.1 MFCC Feature Extraction

**Parameters:**
- Sample rate: 22050 Hz
- Frame length: 0.025s (551 samples)
- Hop length: 0.01s (220 samples)
- MFCC coefficients: 13
- Delta and delta-delta: optional (39 total features)

**Processing:**
1. Load audio file
2. Resample to 22050 Hz if needed
3. Extract MFCC features frame by frame
4. Aggregate (mean, std) across frames
5. Normalize features

### 5.2 Emotion Classification Model

**Architecture Options:**

**Option A: Simple Classifier (scikit-learn)**
- Input: Aggregated MFCC features (13 or 39 dims)
- Model: Random Forest / SVM
- Output: Emotion label + confidence

**Option B: Neural Network (TensorFlow/Keras)**
- Input: Sequence of MFCC frames
- Architecture: LSTM or 1D CNN
- Output: Emotion probabilities (7 classes)

**Training Data:**
- Use public emotion datasets (RAVDESS, CREMA-D)
- Train on MFCC features extracted from labeled audio

**Inference:**
- Load pre-trained model
- Extract MFCC from input audio
- Predict emotion probabilities
- Return top emotion + confidence

## 6. API Design

### 6.1 Request/Response Patterns

**Standard Success Response:**
```json
{
  "success": true,
  "data": {...},
  "message": "Operation successful"
}
```

**Standard Error Response:**
```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message"
  }
}
```

### 6.2 Error Codes

- `AUTH_001`: Invalid credentials
- `AUTH_002`: Token expired
- `AUTH_003`: Token invalid
- `AUDIO_001`: Invalid file format
- `AUDIO_002`: File too large
- `AUDIO_003`: Processing failed
- `RAG_001`: Query processing failed
- `DB_001`: Database connection error
- `VALIDATION_001`: Invalid input data

## 7. Security Design

### 7.1 Authentication Security

- JWT tokens signed with HS256
- Secret key stored in environment variable
- Token expiration: 24 hours
- Password hashing: bcrypt (12 rounds)
- No password storage in plain text

### 7.2 Input Validation

- File type validation (whitelist)
- File size limits
- Email format validation
- SQL injection prevention (MongoDB driver handles)
- XSS prevention (input sanitization)

### 7.3 Data Isolation Security

- User_id extracted from JWT (cannot be spoofed)
- Service layer enforces user_id in all operations
- Database queries parameterized
- No user-controlled collection names

## 8. Performance Optimization

### 8.1 Audio Processing

- Async processing for long operations
- Background tasks for heavy computations
- Temporary file cleanup
- Streaming for large files

### 8.2 Vector Search

- Index optimization in Qdrant
- Batch embedding generation
- Caching frequently accessed collections
- Limit search scope to user collection

### 8.3 Database Queries

- Indexed fields for fast lookups
- Connection pooling
- Query optimization (projection, limit)
- Aggregation pipelines for analytics

## 9. Deployment Considerations

### 9.1 Environment Setup

- Python virtual environment
- MongoDB instance (local or cloud)
- Qdrant server (Docker or standalone)
- Ollama server (local installation)
- Environment variables configuration

### 9.2 File Storage

- Local filesystem for development
- Cloud storage (S3) for production
- Temporary file cleanup job
- User directory structure

### 9.3 Monitoring & Logging

- Structured logging (JSON format)
- Error tracking
- Performance metrics
- API request/response logging

## 10. Testing Strategy

### 10.1 Unit Tests

- Service layer functions
- Utility functions
- Model validation

### 10.2 Integration Tests

- API endpoints
- Database operations
- Authentication flow

### 10.3 End-to-End Tests

- Complete audio processing pipeline
- RAG query flow
- Multi-tenant isolation

