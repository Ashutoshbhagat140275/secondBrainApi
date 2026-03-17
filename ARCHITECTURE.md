# System Architecture Documentation
## Audio Emotion Recognition API with Personalized Learning

**Version**: 1.0  
**Last Updated**: March 6, 2026  
**Architecture Type**: Layered Microservices with Dual-Head ML Pipeline

---

## Table of Contents

1. [High-Level System Architecture](#1-high-level-system-architecture)
2. [Layer Architecture](#2-layer-architecture)
3. [Component Diagrams](#3-component-diagrams)
4. [Data Flow Diagrams](#4-data-flow-diagrams)
5. [Database Schema](#5-database-schema)
6. [ML Pipeline Architecture](#6-ml-pipeline-architecture)
7. [API Endpoints](#7-api-endpoints)
8. [Deployment Architecture](#8-deployment-architecture)

---

## 1. High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           CLIENT LAYER                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │
│  │   Web App    │  │  Mobile App  │  │  API Client  │                  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                  │
└─────────┼──────────────────┼──────────────────┼──────────────────────────┘
          │                  │                  │
          └──────────────────┴──────────────────┘
                             │
                    ┌────────▼────────┐
                    │   CORS/Auth     │
                    │   Middleware    │
                    └────────┬────────┘
                             │
┌────────────────────────────┼────────────────────────────────────────────┐
│                    API GATEWAY LAYER (FastAPI)                          │
│  ┌─────────────────────────▼──────────────────────────────┐            │
│  │              FastAPI Application (main.py)              │            │
│  │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐│            │
│  │  │ Auth │ │Audio │ │ RAG  │ │Dashb.│ │Admin │ │Feedb.││            │
│  │  │Router│ │Router│ │Router│ │Router│ │Router│ │Router││            │
│  │  └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘│            │
│  └─────┼────────┼────────┼────────┼────────┼────────┼─────┘            │
└────────┼────────┼────────┼────────┼────────┼────────┼──────────────────┘
         │        │        │        │        │        │
┌────────┼────────┼────────┼────────┼────────┼────────┼──────────────────┐
│        │   SERVICE/BUSINESS LOGIC LAYER                                 │
│  ┌─────▼──┐ ┌─▼────────┐ ┌─▼──────┐ ┌─▼────────┐ ┌─▼────────┐         │
│  │  Auth  │ │  Audio   │ │  RAG   │ │Dashboard │ │  Admin   │         │
│  │Service │ │Processor │ │Service │ │ Service  │ │ Service  │         │
│  └────────┘ └─┬────────┘ └────────┘ └──────────┘ └──────────┘         │
│               │                                                         │
│  ┌────────────▼──────────────────────────────────────────┐             │
│  │         ML PIPELINE (Dual-Head Architecture)          │             │
│  │  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐ │             │
│  │  │  Wav2Vec2    │  │  Dual-Head   │  │   Alpha     │ │             │
│  │  │   Encoder    │→ │  Classifier  │→ │   Engine    │ │             │
│  │  └──────────────┘  └──────┬───────┘  └─────────────┘ │             │
│  │                            │                           │             │
│  │         ┌──────────────────┴──────────────────┐       │             │
│  │         │                                      │       │             │
│  │  ┌──────▼──────┐                    ┌─────────▼─────┐ │             │
│  │  │   Global    │                    │  User Heads   │ │             │
│  │  │    Head     │                    │  (Per-User)   │ │             │
│  │  │  (Trained)  │                    │  (Lazy Load)  │ │             │
│  │  └─────────────┘                    └───────────────┘ │             │
│  └────────────────────────────────────────────────────────┘             │
│                                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                 │
│  │  Feedback    │  │  Training    │  │   Vector     │                 │
│  │  Service     │  │  Job Tracker │  │   Store      │                 │
│  └──────────────┘  └──────────────┘  └──────────────┘                 │
└──────────────────────────────────────────────────────────────────────────┘
                             │
┌────────────────────────────┼────────────────────────────────────────────┐
│                    DATA/STORAGE LAYER                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │
│  │   MongoDB   │  │   Qdrant    │  │    Redis    │  │  File Sys   │  │
│  │  (Primary)  │  │  (Vectors)  │  │   (Cache)   │  │  (Models)   │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │
│                                                                          │
│  Collections:      Collections:      Keys:            Directories:      │
│  • users           • user_*_docs    • query_cache    • models/         │
│  • audio_sessions  (per-user)       • user_cache     • uploads/        │
│  • emotions                                           • user_heads/     │
│  • user_feedback                                                        │
│  • training_jobs                                                        │
│  • user_models                                                          │
└──────────────────────────────────────────────────────────────────────────┘
                             │
┌────────────────────────────┼────────────────────────────────────────────┐
│                    EXTERNAL SERVICES                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                    │
│  │  Whisper    │  │   Ollama    │  │ HuggingFace │                    │
│  │(Transcribe) │  │  (LLM/RAG)  │  │  (Models)   │                    │
│  └─────────────┘  └─────────────┘  └─────────────┘                    │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Layer Architecture

### 2.1 Presentation Layer (API Gateway)

**Technology**: FastAPI  
**Responsibility**: HTTP request handling, routing, authentication, CORS

**Components**:
- `main.py` - Application entry point, middleware configuration
- `routers/` - API endpoint definitions
  - `auth.py` - User registration, login, JWT token management
  - `audio.py` - Audio upload and emotion classification
  - `feedback.py` - User feedback submission
  - `rag.py` - RAG query processing
  - `dashboard.py` - User statistics and history
  - `admin.py` - Admin operations (model management, cleanup)

**Key Features**:
- JWT-based authentication
- CORS middleware for cross-origin requests
- Request validation using Pydantic schemas
- Automatic API documentation (Swagger/OpenAPI)

### 2.2 Business Logic Layer (Services)

**Responsibility**: Core business logic, ML inference, data processing

**Service Components**:

1. **Authentication Service** (`services/auth.py`)
   - Password hashing (bcrypt)
   - JWT token generation/validation
   - User session management

2. **Audio Processing Pipeline** (`services/audio_processor.py`)
   - File validation and storage
   - Audio preprocessing (16kHz resampling, VAD, normalization)
   - Orchestrates ML pipeline
   - Metadata persistence

3. **ML Pipeline Services**:
   - `wav2vec2_encoder.py` - Extract 768-dim embeddings
   - `dual_head_classifier.py` - Orchestrate dual-head inference
   - `global_emotion_head.py` - Global model predictions
   - `user_emotion_head.py` - User-specific model predictions (LRU cached)
   - `alpha_engine.py` - Adaptive blending weight computation

4. **Feedback Service** (`services/feedback_service.py`)
   - Feedback validation and storage
   - Training trigger logic
   - Task queue integration

5. **Training Job Tracker** (`services/training_job_tracker.py`)
   - Job lifecycle management
   - Status tracking (queued → running → completed/failed)
   - Metrics storage

6. **RAG Service** (`services/rag_service.py`)
   - Vector search in Qdrant
   - LLM query generation (Ollama)
   - Answer synthesis with source attribution

7. **Vector Store** (`services/vector_store.py`)
   - Qdrant collection management
   - Document embedding and storage
   - Semantic search

8. **Query Cache** (`services/query_cache.py`)
   - Redis-based caching
   - Semantic similarity matching
   - Cache invalidation

### 2.3 Data Access Layer

**Responsibility**: Database connections, data persistence, model storage

**Components**:


1. **MongoDB** (`db/mongodb.py`)
   - Primary data store
   - Collections: users, audio_sessions, emotions, user_feedback, training_jobs, user_models
   - Indexes for performance optimization

2. **Qdrant** (`db/qdrant.py`)
   - Vector database for semantic search
   - Per-user collections for data isolation
   - Cosine similarity search

3. **Redis** (`db/redis.py`)
   - Query result caching
   - User model caching (LRU)
   - Session management

4. **File System**
   - Audio uploads: `./uploads/{user_id}/`
   - ML models: `./models/`
   - User heads: `./models/user_heads/{user_id}.pt`

### 2.4 External Services Layer

**Responsibility**: Integration with third-party services

**Services**:
1. **Whisper** - Audio transcription (OpenAI Whisper)
2. **Ollama** - LLM for RAG answer generation
3. **HuggingFace** - Pre-trained model downloads (Wav2Vec2, embeddings)

---

## 3. Component Diagrams

### 3.1 Audio Processing Component

```
┌─────────────────────────────────────────────────────────────────┐
│                    Audio Processing Pipeline                     │
│                                                                   │
│  Input: Audio File (WAV/MP3/M4A/FLAC/OGG)                       │
│                                                                   │
│  ┌────────────────┐                                              │
│  │  1. Validate   │  Check format, size, authentication         │
│  └───────┬────────┘                                              │
│          │                                                        │
│  ┌───────▼────────┐                                              │
│  │  2. Save File  │  Store in ./uploads/{user_id}/              │
│  └───────┬────────┘                                              │
│          │                                                        │
│  ┌───────▼────────┐                                              │
│  │ 3. Preprocess  │  • Resample to 16kHz                        │
│  │                │  • Voice Activity Detection (VAD)           │
│  │                │  • Peak normalization [-1, 1]               │
│  └───────┬────────┘                                              │
│          │                                                        │
│  ┌───────▼────────┐                                              │
│  │ 4. Extract     │  Wav2Vec2 → 768-dim embedding               │
│  │    Embedding   │  (~500ms latency)                           │
│  └───────┬────────┘                                              │
│          │                                                        │
│  ┌───────▼────────┐                                              │
│  │ 5. Classify    │  Dual-Head Classifier                       │
│  │    Emotion     │  • Global head prediction                   │
│  │                │  • User head prediction (if available)      │
│  │                │  • Alpha engine blending                    │
│  └───────┬────────┘                                              │
│          │                                                        │
│  ┌───────▼────────┐                                              │
│  │ 6. Transcribe  │  Whisper → text transcription               │
│  │                │  (~2-3s latency)                            │
│  └───────┬────────┘                                              │
│          │                                                        │
│  ┌───────▼────────┐                                              │
│  │ 7. Store Data  │  • MongoDB: session, emotion                │
│  │                │  • Qdrant: vector embedding                 │
│  │                │  • Invalidate cache                         │
│  └───────┬────────┘                                              │
│          │                                                        │
│  Output: {session_id, emotion, confidence, transcription, ...}  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Dual-Head Classification Component

```
┌─────────────────────────────────────────────────────────────────┐
│              Dual-Head Emotion Classifier                        │
│                                                                   │
│  Input: 768-dim Wav2Vec2 Embedding                              │
│                                                                   │
│  ┌────────────────────────────────────────────────────────┐     │
│  │              Step 1: Global Head Prediction            │     │
│  │  ┌──────────────────────────────────────────────────┐  │     │
│  │  │  Global Head (global_emotion_head.pt)            │  │     │
│  │  │  • Trained on RAVDESS + CREMA-D                  │  │     │
│  │  │  • 768 → 8 emotions                              │  │     │
│  │  │  • Always available (eager loaded)               │  │     │
│  │  └──────────────────┬───────────────────────────────┘  │     │
│  │                     │                                    │     │
│  │                     ▼                                    │     │
│  │         P_g = [p1, p2, ..., p8]  (probabilities)        │     │
│  │         C_g = max(P_g)           (confidence)           │     │
│  └────────────────────┬───────────────────────────────────┘     │
│                       │                                          │
│  ┌────────────────────▼───────────────────────────────────┐     │
│  │              Step 2: User Head Prediction              │     │
│  │  ┌──────────────────────────────────────────────────┐  │     │
│  │  │  User Head (user_heads/{user_id}.pt)            │  │     │
│  │  │  • Trained on user feedback                      │  │     │
│  │  │  • 768 → 8 emotions                              │  │     │
│  │  │  • Lazy loaded (LRU cache, max 100 users)       │  │     │
│  │  │  • Returns None if not trained yet              │  │     │
│  │  └──────────────────┬───────────────────────────────┘  │     │
│  │                     │                                    │     │
│  │                     ▼                                    │     │
│  │         P_u = [p1, p2, ..., p8]  (or None)              │     │
│  │         C_u = max(P_u)           (or None)              │     │
│  └────────────────────┬───────────────────────────────────┘     │
│                       │                                          │
│  ┌────────────────────▼───────────────────────────────────┐     │
│  │           Step 3: Alpha Engine (Blending Weight)       │     │
│  │  ┌──────────────────────────────────────────────────┐  │     │
│  │  │  Sigmoid Formula (default):                      │  │     │
│  │  │    α_data = 1 / (1 + N/K)                        │  │     │
│  │  │    α_conf = 1 / (1 + exp(-β(C_g - τ)))          │  │     │
│  │  │    α = α_data × α_conf                           │  │     │
│  │  │                                                   │  │     │
│  │  │  Where:                                           │  │     │
│  │  │    N = feedback_count                            │  │     │
│  │  │    K = 50 (feedback scale)                       │  │     │
│  │  │    τ = 0.6 (confidence threshold)                │  │     │
│  │  │    β = 10 (sigmoid sharpness)                    │  │     │
│  │  └──────────────────┬───────────────────────────────┘  │     │
│  │                     │                                    │     │
│  │                     ▼                                    │     │
│  │                  α ∈ (0, 1]                              │     │
│  └────────────────────┬───────────────────────────────────┘     │
│                       │                                          │
│  ┌────────────────────▼───────────────────────────────────┐     │
│  │              Step 4: Blend Predictions                 │     │
│  │                                                          │     │
│  │  If P_u exists:                                         │     │
│  │    P_f = α·P_g + (1-α)·P_u                             │     │
│  │    P_f = P_f / sum(P_f)  (normalize)                   │     │
│  │  Else:                                                  │     │
│  │    P_f = P_g  (global only, α=1.0)                     │     │
│  │                                                          │     │
│  │  Final emotion = argmax(P_f)                            │     │
│  │  Final confidence = max(P_f)                            │     │
│  └─────────────────────────────────────────────────────────┘     │
│                                                                   │
│  Output: {emotion, confidence, global_*, user_*, blend_weight}  │
└─────────────────────────────────────────────────────────────────┘
```



### 3.3 Feedback & Training Component

```
┌─────────────────────────────────────────────────────────────────┐
│              Feedback & Training Pipeline                        │
│                                                                   │
│  ┌────────────────────────────────────────────────────────┐     │
│  │         User Submits Feedback Correction              │     │
│  │  POST /feedback                                         │     │
│  │  {session_id, corrected_emotion}                       │     │
│  └────────────────────┬───────────────────────────────────┘     │
│                       │                                          │
│  ┌────────────────────▼───────────────────────────────────┐     │
│  │         Step 1: Validate & Retrieve Data               │     │
│  │  • Verify session ownership                            │     │
│  │  • Validate emotion label                              │     │
│  │  • Retrieve embedding from EmotionAnalysis             │     │
│  │  • Get predicted emotion from AudioSession             │     │
│  └────────────────────┬───────────────────────────────────┘     │
│                       │                                          │
│  ┌────────────────────▼───────────────────────────────────┐     │
│  │         Step 2: Store Feedback                         │     │
│  │  MongoDB.user_feedback.insert({                        │     │
│  │    user_id, session_id, embedding,                     │     │
│    predicted_emotion, corrected_emotion, timestamp       │     │
│  │  })                                                     │     │
│  └────────────────────┬───────────────────────────────────┘     │
│                       │                                          │
│  ┌────────────────────▼───────────────────────────────────┐     │
│  │         Step 3: Increment Feedback Count               │     │
│  │  User.increment_feedback_count(user_id)                │     │
│  │  → feedback_count++                                     │     │
│  └────────────────────┬───────────────────────────────────┘     │
│                       │                                          │
│  ┌────────────────────▼───────────────────────────────────┐     │
│  │         Step 4: Check Training Trigger                 │     │
│  │                                                          │     │
│  │  Trigger if:                                            │     │
│  │    feedback_count >= 20 AND                            │     │
│  │    feedback_count % 10 == 0                            │     │
│  │                                                          │     │
│  │  Training Schedule:                                     │     │
│  │    • 20 samples: Initial training                      │     │
│  │    • 30 samples: 1st incremental                       │     │
│  │    • 40 samples: 2nd incremental                       │     │
│  │    • 50 samples: 3rd incremental                       │     │
│  │    • ... every 10 samples                              │     │
│  └────────────────────┬───────────────────────────────────┘     │
│                       │                                          │
│                       │ (if triggered)                           │
│                       ▼                                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │         Step 5: Enqueue Training Job                    │    │
│  │  • Create TrainingJob record (status=queued)           │    │
│  │  • Add to TaskQueue                                     │    │
│  │  • Return job_id to user                                │    │
│  └────────────────────┬────────────────────────────────────┘    │
│                       │                                          │
│                       ▼                                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │         Background: Training Worker                     │    │
│  │  ┌───────────────────────────────────────────────────┐  │    │
│  │  │  1. Update status → running                       │  │    │
│  │  │  2. Fetch feedback samples from MongoDB           │  │    │
│  │  │  3. Prepare dataset (embeddings + labels)         │  │    │
│  │  │  4. Train user head (PyTorch)                     │  │    │
│  │  │     • Architecture: Linear(768 → 8)               │  │    │
│  │  │     • Loss: CrossEntropyLoss                      │  │    │
│  │  │     • Optimizer: Adam (lr=0.001)                  │  │    │
│  │  │     • Epochs: 50                                  │  │    │
│  │  │  5. Save model → user_heads/{user_id}.pt          │  │    │
│  │  │  6. Invalidate user cache                         │  │    │
│  │  │  7. Update status → completed (with metrics)      │  │    │
│  │  └───────────────────────────────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                   │
│  Output: {status, feedback_count, training_triggered, job_id}   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Data Flow Diagrams

### 4.1 Audio Upload & Classification Flow

```
User → API → Audio Processor → Wav2Vec2 → Dual-Head → MongoDB/Qdrant
                                                ↓
                                          Alpha Engine
                                                ↓
                                    Global Head + User Head
                                                ↓
                                        Blended Prediction
```

**Detailed Flow**:

1. **User uploads audio** → `POST /audio/upload`
2. **API validates** → File format, size, authentication
3. **Audio Processor**:
   - Saves file to disk
   - Preprocesses (16kHz, VAD, normalize)
4. **Wav2Vec2 Encoder** → Extracts 768-dim embedding
5. **Dual-Head Classifier**:
   - Global head predicts → P_g, C_g
   - User head predicts → P_u, C_u (if available)
   - Alpha engine computes → α
   - Blends predictions → P_f
6. **Whisper** → Transcribes audio to text
7. **Storage**:
   - MongoDB: AudioSession, EmotionAnalysis
   - Qdrant: Vector embedding + metadata
8. **Response** → {session_id, emotion, confidence, transcription, ...}

**Latency Breakdown**:
- File upload: ~100ms
- Preprocessing: ~200ms
- Wav2Vec2: ~500ms
- Classification: ~5ms
- Transcription: ~2-3s
- Storage: ~100ms
- **Total**: ~3-4 seconds

### 4.2 Feedback & Training Flow

```
User → Feedback API → Validate → Store → Increment Count → Check Trigger
                                                                  ↓
                                                            (if triggered)
                                                                  ↓
                                                          Create Training Job
                                                                  ↓
                                                            Task Queue
                                                                  ↓
                                                          Training Worker
                                                                  ↓
                                                    Train User Head (PyTorch)
                                                                  ↓
                                                      Save Model + Invalidate Cache
```

**Detailed Flow**:

1. **User submits feedback** → `POST /feedback`
2. **Validate**:
   - Session exists and belongs to user
   - Emotion label is valid
3. **Retrieve data**:
   - Embedding from EmotionAnalysis
   - Predicted emotion from AudioSession
4. **Store feedback** → MongoDB.user_feedback
5. **Increment count** → User.feedback_count++
6. **Check trigger**:
   - If count >= 20 AND count % 10 == 0 → trigger training
7. **Create job** → TrainingJob (status=queued)
8. **Enqueue** → TaskQueue.enqueue(train_user_head)
9. **Background worker**:
   - Fetch feedback samples
   - Train PyTorch model
   - Save to user_heads/{user_id}.pt
   - Update job status → completed

**Training Time**: ~5-8 seconds for 20-100 samples



### 4.3 RAG Query Flow

```
User → RAG API → Vector Search (Qdrant) → LLM (Ollama) → Response
                        ↓
                  Check Cache (Redis)
                        ↓
                  (if cache miss)
                        ↓
                Semantic Search
                        ↓
                Retrieve Documents
                        ↓
                Generate Answer
                        ↓
                Store in Cache
```

**Detailed Flow**:

1. **User submits query** → `POST /rag/query`
2. **Check cache** → Redis (semantic similarity > 0.95)
3. **If cache miss**:
   - Embed query using sentence-transformers
   - Search Qdrant for top-k similar documents
   - Build context from retrieved documents
   - Send to Ollama LLM for answer generation
   - Store result in cache
4. **Response** → {answer, sources, confidence}

**Latency**: ~1-2 seconds (cached: ~50ms)

---

## 5. Database Schema

### 5.1 MongoDB Collections

#### Collection: `users`
```javascript
{
  _id: ObjectId,
  email: String (unique),
  password_hash: String (bcrypt),
  feedback_count: Integer (default: 0),
  is_admin: Boolean (default: false),
  created_at: DateTime
}
```
**Indexes**: email (unique)

#### Collection: `audio_sessions`
```javascript
{
  _id: ObjectId,
  user_id: String,
  audio_file_path: String,
  emotion_data: {
    emotion: String,
    confidence: Float
  },
  transcription_text: String,
  qdrant_collection_id: String,
  timestamp: DateTime
}
```
**Indexes**: user_id, timestamp

#### Collection: `emotions`
```javascript
{
  _id: ObjectId,
  user_id: String,
  session_id: String,
  emotion_label: String,
  confidence: Float,
  mfcc_features: Array[Float] (768-dim embedding),
  timestamp: DateTime
}
```
**Indexes**: user_id, session_id

#### Collection: `user_feedback`
```javascript
{
  _id: ObjectId,
  user_id: String,
  session_id: String,
  embedding: Array[Float] (768-dim),
  predicted_emotion: String,
  corrected_emotion: String,
  timestamp: DateTime
}
```
**Indexes**: user_id, timestamp, (user_id, timestamp)

#### Collection: `training_jobs`
```javascript
{
  _id: ObjectId,
  user_id: String,
  job_id: String (UUID, unique),
  status: String (queued|running|completed|failed),
  created_at: DateTime,
  started_at: DateTime (nullable),
  completed_at: DateTime (nullable),
  updated_at: DateTime,
  error_message: String (nullable),
  metrics: {
    loss: Float,
    accuracy: Float,
    samples_count: Integer
  } (nullable)
}
```
**Indexes**: user_id, job_id (unique), (user_id, created_at DESC)

#### Collection: `user_models`
```javascript
{
  _id: ObjectId,
  user_id: String (unique),
  model_data: Binary (gzip compressed PyTorch state_dict),
  model_size_bytes: Integer,
  feedback_count: Integer,
  created_at: DateTime,
  updated_at: DateTime
}
```
**Indexes**: user_id (unique), updated_at

### 5.2 Qdrant Collections

#### Collection: `user_{user_id}_documents`
```javascript
{
  id: UUID,
  vector: Array[Float] (384-dim sentence embedding),
  payload: {
    user_id: String,
    text: String (transcription),
    session_id: String,
    timestamp: String (ISO format),
    emotion_label: String
  }
}
```
**Vector Config**: 
- Size: 384
- Distance: Cosine
- Per-user isolation

### 5.3 Redis Keys

#### Key Pattern: `query_cache:{user_id}:{query_hash}`
```
Value: JSON {
  answer: String,
  sources: Array[Object],
  timestamp: Float
}
TTL: 3600 seconds (1 hour)
```

#### Key Pattern: `user_model_cache:{user_id}`
```
Value: Binary (pickled PyTorch model)
TTL: None (LRU eviction, max 100 keys)
```

---

## 6. ML Pipeline Architecture

### 6.1 Model Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Dual-Head Architecture                        │
│                                                                   │
│  ┌────────────────────────────────────────────────────────┐     │
│  │              Wav2Vec2 Encoder (Frozen)                 │     │
│  │  • Model: facebook/wav2vec2-base-960h                  │     │
│  │  • Input: Audio waveform (16kHz)                       │     │
│  │  • Output: 768-dim embedding                           │     │
│  │  • Pre-trained on 960h LibriSpeech                     │     │
│  └────────────────────┬───────────────────────────────────┘     │
│                       │                                          │
│                       │ 768-dim embedding                        │
│                       │                                          │
│         ┌─────────────┴─────────────┐                           │
│         │                           │                           │
│  ┌──────▼──────┐             ┌──────▼──────┐                   │
│  │ Global Head │             │  User Head  │                   │
│  │             │             │  (Per-User) │                   │
│  │ Linear(768→8)│             │ Linear(768→8)│                   │
│  │             │             │             │                   │
│  │ Trained on: │             │ Trained on: │                   │
│  │ • RAVDESS   │             │ • User      │                   │
│  │ • CREMA-D   │             │   Feedback  │                   │
│  │             │             │             │                   │
│  │ 72.7% acc   │             │ Variable    │                   │
│  └──────┬──────┘             └──────┬──────┘                   │
│         │                           │                           │
│         │ P_g, C_g                  │ P_u, C_u                  │
│         │                           │                           │
│         └─────────────┬─────────────┘                           │
│                       │                                          │
│                ┌──────▼──────┐                                  │
│                │ Alpha Engine│                                  │
│                │   α = f(N,  │                                  │
│                │      C_g)   │                                  │
│                └──────┬──────┘                                  │
│                       │                                          │
│                ┌──────▼──────┐                                  │
│                │   Blending  │                                  │
│                │ P_f = α·P_g │                                  │
│                │  + (1-α)·P_u│                                  │
│                └──────┬──────┘                                  │
│                       │                                          │
│                       ▼                                          │
│              Final Prediction                                    │
│              (8 emotion classes)                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Emotion Classes

```
1. happy
2. sad
3. angry
4. fearful
5. disgusted
6. surprised
7. neutral
8. calm
```

### 6.3 Training Datasets

**Global Head Training**:
- RAVDESS: ~1,440 audio files (24 actors, 8 emotions)
- CREMA-D: ~7,442 audio files (91 actors, 6 emotions)
- Total: ~8,882 samples
- Accuracy: 72.7%

**User Head Training**:
- User feedback corrections
- Minimum: 20 samples
- Incremental updates: Every 10 samples
- Training time: ~5-8 seconds

### 6.4 Alpha Engine Formulas

**Sigmoid Formula** (default):
```
α_data = 1 / (1 + N/K)
α_conf = 1 / (1 + exp(-β(C_g - τ)))
α = α_data × α_conf

Parameters:
  K = 50    (feedback scale)
  τ = 0.6   (confidence threshold)
  β = 10    (sigmoid sharpness)
```

**Linear Formula** (legacy):
```
α = 0.5 + 0.3·C_g - 0.2·min(N/100, 1.0)
α = clamp(α, 0.3, 1.0)

Special case: N < 20 → α = 1.0
```

---

## 7. API Endpoints

### 7.1 Authentication Endpoints

```
POST   /auth/register          Register new user
POST   /auth/login             Login and get JWT token
GET    /auth/me                Get current user info
```

### 7.2 Audio Endpoints

```
POST   /audio/upload           Upload audio for emotion analysis
GET    /audio/sessions         Get user's audio sessions
GET    /audio/sessions/{id}    Get specific session details
```

### 7.3 Feedback Endpoints

```
POST   /feedback               Submit emotion correction
GET    /feedback/history       Get user's feedback history
```

### 7.4 Dashboard Endpoints

```
GET    /dashboard/emotions     Get emotion history (with filters)
GET    /dashboard/stats        Get user statistics
GET    /dashboard/training     Get training job status
```

### 7.5 RAG Endpoints

```
POST   /rag/query              Submit RAG query
GET    /rag/history            Get query history
```

### 7.6 Admin Endpoints

```
GET    /admin/models           List all user models
POST   /admin/models/cleanup   Cleanup inactive models
POST   /admin/training/trigger Manually trigger training
GET    /admin/users            List all users
```

---

## 8. Deployment Architecture



### 8.1 Production Deployment

```
┌─────────────────────────────────────────────────────────────────┐
│                         Load Balancer                            │
│                      (NGINX / AWS ALB)                           │
└────────────────────────┬────────────────────────────────────────┘
                         │
         ┌───────────────┴───────────────┐
         │                               │
┌────────▼────────┐             ┌────────▼────────┐
│  FastAPI App 1  │             │  FastAPI App 2  │
│  (Gunicorn)     │             │  (Gunicorn)     │
│  Port: 8000     │             │  Port: 8001     │
└────────┬────────┘             └────────┬────────┘
         │                               │
         └───────────────┬───────────────┘
                         │
         ┌───────────────┴───────────────┐
         │                               │
         │                               │
┌────────▼────────┐  ┌────────▼────────┐  ┌────────▼────────┐
│    MongoDB      │  │     Qdrant      │  │      Redis      │
│   (Replica Set) │  │   (Clustered)   │  │   (Sentinel)    │
└─────────────────┘  └─────────────────┘  └─────────────────┘
         │                   │                     │
         └───────────────────┴─────────────────────┘
                             │
                    ┌────────▼────────┐
                    │  Shared Storage │
                    │  (NFS / S3)     │
                    │  • Models       │
                    │  • Audio Files  │
                    └─────────────────┘
```

### 8.2 Container Architecture (Docker)

```yaml
services:
  api:
    image: emotion-api:latest
    ports: ["8000:8000"]
    environment:
      - MONGODB_URL=mongodb://mongo:27017
      - QDRANT_URL=http://qdrant:6333
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./models:/app/models
      - ./uploads:/app/uploads
    depends_on: [mongo, qdrant, redis]
  
  mongo:
    image: mongo:6.0
    ports: ["27017:27017"]
    volumes: [mongo_data:/data/db]
  
  qdrant:
    image: qdrant/qdrant:latest
    ports: ["6333:6333"]
    volumes: [qdrant_data:/qdrant/storage]
  
  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
    volumes: [redis_data:/data]
  
  worker:
    image: emotion-api:latest
    command: python -m training.worker
    environment: [same as api]
    volumes: [same as api]
```

### 8.3 Scalability Considerations

**Horizontal Scaling**:
- Multiple FastAPI instances behind load balancer
- Stateless API design (JWT tokens, no server sessions)
- Shared storage for models and audio files

**Vertical Scaling**:
- GPU support for Wav2Vec2 inference (optional)
- Increased memory for model caching
- SSD storage for faster model loading

**Database Scaling**:
- MongoDB replica set for read scaling
- Qdrant clustering for vector search
- Redis Sentinel for high availability

**Bottlenecks**:
1. Wav2Vec2 inference (~500ms) - Consider GPU or model quantization
2. Whisper transcription (~2-3s) - Consider async processing
3. User head training (~5-8s) - Already async via task queue

---

## 9. Security Architecture

### 9.1 Authentication Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    Authentication Flow                           │
│                                                                   │
│  1. User Registration                                            │
│     POST /auth/register {email, password}                        │
│          ↓                                                        │
│     Validate email format                                        │
│          ↓                                                        │
│     Hash password (bcrypt, cost=12)                              │
│          ↓                                                        │
│     Store in MongoDB                                             │
│          ↓                                                        │
│     Return {user_id, email}                                      │
│                                                                   │
│  2. User Login                                                   │
│     POST /auth/login {email, password}                           │
│          ↓                                                        │
│     Fetch user from MongoDB                                      │
│          ↓                                                        │
│     Verify password (bcrypt.checkpw)                             │
│          ↓                                                        │
│     Generate JWT token                                           │
│       • Payload: {user_id, email, is_admin}                      │
│       • Algorithm: HS256                                         │
│       • Expiration: 24 hours                                     │
│       • Secret: From environment variable                        │
│          ↓                                                        │
│     Return {access_token, token_type: "bearer"}                  │
│                                                                   │
│  3. Protected Endpoint Access                                    │
│     GET /audio/sessions                                          │
│     Headers: {Authorization: "Bearer <token>"}                   │
│          ↓                                                        │
│     Extract token from header                                    │
│          ↓                                                        │
│     Verify JWT signature                                         │
│          ↓                                                        │
│     Check expiration                                             │
│          ↓                                                        │
│     Extract user_id from payload                                 │
│          ↓                                                        │
│     Inject into request context                                  │
│          ↓                                                        │
│     Process request with user_id                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 9.2 Authorization Model

**Role-Based Access Control (RBAC)**:

```
User Roles:
  • Regular User (is_admin=false)
    - Upload audio
    - Submit feedback
    - View own data
    - Query own RAG data
  
  • Admin User (is_admin=true)
    - All regular user permissions
    - View all user models
    - Trigger training for any user
    - Cleanup models
    - View system statistics
```

**Data Isolation**:
- Users can only access their own audio sessions
- Users can only submit feedback for their own sessions
- Qdrant collections are per-user (user_{user_id}_documents)
- User heads are stored separately (user_heads/{user_id}.pt)

### 9.3 Security Best Practices

**Input Validation**:
- Pydantic schemas for request validation
- File type validation (audio formats only)
- File size limits (50MB default)
- Emotion label validation (must be in EMOTION_LABELS)

**Data Protection**:
- Passwords hashed with bcrypt (cost=12)
- JWT tokens signed with secret key
- HTTPS enforcement (production)
- CORS configuration (restrict origins in production)

**Error Handling**:
- No sensitive data in error messages
- Generic error responses for authentication failures
- Logging without exposing credentials

---

## 10. Monitoring & Observability

### 10.1 Logging Strategy

```python
# Log Levels:
DEBUG   - Detailed diagnostic info (alpha computation, cache hits)
INFO    - General system events (model loaded, training triggered)
WARNING - Recoverable issues (service unavailable, fallback used)
ERROR   - Serious problems (database connection failed)

# Log Format:
"%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Key Log Points:
- API request/response (with latency)
- Model loading/caching
- Training job lifecycle
- Database operations
- External service calls
```

### 10.2 Metrics to Monitor

**Application Metrics**:
- Request rate (req/s)
- Response latency (p50, p95, p99)
- Error rate (4xx, 5xx)
- Active users
- Feedback submission rate
- Training job queue length

**ML Metrics**:
- Wav2Vec2 inference latency
- Classification latency
- Global head confidence distribution
- User head availability rate
- Alpha distribution
- Training success rate

**Infrastructure Metrics**:
- CPU usage
- Memory usage
- Disk I/O
- Network I/O
- Database connections
- Cache hit rate

### 10.3 Health Checks

```python
GET /health
Response: {"status": "healthy"}

GET /health/detailed
Response: {
  "status": "healthy",
  "services": {
    "mongodb": "connected",
    "qdrant": "connected",
    "redis": "connected",
    "global_head": "loaded",
    "user_heads_cached": 42
  },
  "uptime": 86400,
  "version": "1.0.0"
}
```

---

## 11. Performance Optimization

### 11.1 Caching Strategy

**Model Caching**:
- Global head: Eager loaded at startup (always in memory)
- User heads: Lazy loaded with LRU cache (max 100 users)
- Cache invalidation: On training completion

**Query Caching**:
- Redis-based semantic cache
- Similarity threshold: 0.95 (cosine)
- TTL: 1 hour
- Per-user isolation

**Database Indexing**:
- MongoDB: user_id, timestamp, (user_id, timestamp)
- Qdrant: Vector index (HNSW)

### 11.2 Async Processing

**Background Tasks**:
- User head training (via task queue)
- Model cleanup (scheduled jobs)
- Cache warming (optional)

**Async Endpoints**:
- All database operations
- External service calls (Ollama, Whisper)

### 11.3 Resource Management

**Memory Management**:
- LRU cache for user heads (prevents memory bloat)
- Lazy loading (models loaded on-demand)
- Periodic cleanup of old audio files

**Connection Pooling**:
- MongoDB: Connection pool (default: 100)
- Redis: Connection pool (default: 50)
- Qdrant: HTTP connection reuse

---

## 12. Error Handling & Recovery

### 12.1 Error Categories

**Client Errors (4xx)**:
- 400 Bad Request: Invalid input, validation failure
- 401 Unauthorized: Missing/invalid token
- 403 Forbidden: Insufficient permissions
- 404 Not Found: Resource doesn't exist
- 413 Payload Too Large: File size exceeded

**Server Errors (5xx)**:
- 500 Internal Server Error: Unexpected failure
- 503 Service Unavailable: External service down

### 12.2 Fallback Strategies

**Model Unavailable**:
- Global head missing → Uniform distribution (1/8 per class)
- User head missing → Use global head only (α=1.0)

**Service Unavailable**:
- Qdrant down → Skip vector storage, continue processing
- Redis down → Disable caching, direct queries
- Ollama down → Return 503 for RAG queries

**Training Failures**:
- Insufficient data → Job fails with clear error message
- Training error → Job marked as failed, model unchanged
- Retry logic → Manual retry via admin endpoint

---

## 13. Development Workflow

### 13.1 Local Development Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start services (Docker Compose)
docker-compose up -d mongo qdrant redis

# 3. Set environment variables
cp .env.example .env
# Edit .env with local configuration

# 4. Run migrations (create indexes)
python -m app.db.mongodb

# 5. Train global head (if not exists)
python training/train_global_head.py

# 6. Start API server
uvicorn app.main:app --reload --port 8000

# 7. Access API docs
# http://localhost:8000/docs
```

### 13.2 Testing Strategy

**Unit Tests**:
- Service layer logic
- Alpha engine formulas
- Blending functions
- Validation logic

**Integration Tests**:
- API endpoints
- Database operations
- ML pipeline end-to-end

**Property-Based Tests**:
- Alpha engine invariants
- Probability normalization
- Blending correctness

**Performance Tests**:
- Load testing (Locust)
- Latency benchmarks
- Scalability tests

---

## 14. Technology Stack Summary

### 14.1 Backend Framework
- **FastAPI** - Modern async web framework
- **Pydantic** - Data validation
- **Uvicorn** - ASGI server

### 14.2 Machine Learning
- **PyTorch** - Deep learning framework
- **Transformers** - Wav2Vec2 model
- **Whisper** - Audio transcription
- **Sentence-Transformers** - Text embeddings

### 14.3 Databases
- **MongoDB** - Primary data store
- **Qdrant** - Vector database
- **Redis** - Caching layer

### 14.4 External Services
- **Ollama** - LLM for RAG
- **HuggingFace** - Model hub

### 14.5 Development Tools
- **pytest** - Testing framework
- **Hypothesis** - Property-based testing
- **Docker** - Containerization
- **Git** - Version control

---

## 15. Key Design Decisions

### 15.1 Why Dual-Head Architecture?

**Problem**: Single global model doesn't adapt to individual users

**Solution**: Dual-head with adaptive blending
- Global head: Robust baseline (trained on 8,882 samples)
- User head: Personalized (trained on user feedback)
- Alpha engine: Smooth transition based on data and confidence

**Benefits**:
- New users get good predictions immediately (global head)
- Experienced users get personalized predictions
- Graceful degradation (fallback to global if user head fails)

### 15.2 Why Sigmoid Alpha Formula?

**Problem**: Linear formula has abrupt transitions and hard clamping

**Solution**: Sigmoid-based formula with separate components
- α_data: Exponential decay (smooth feedback scaling)
- α_conf: Sigmoid transition (smooth confidence scaling)
- α = α_data × α_conf: Multiplicative (both must agree)

**Benefits**:
- No hard clamping needed (natural bounds from sigmoid)
- Smooth transitions (no discontinuities)
- Interpretable components (data vs confidence)

### 15.3 Why LRU Cache for User Heads?

**Problem**: Loading all user models into memory is infeasible

**Solution**: Lazy loading with LRU eviction (max 100 users)

**Benefits**:
- Memory efficient (only active users cached)
- Fast inference for cached users (~5ms)
- Automatic eviction of inactive users

### 15.4 Why Per-User Qdrant Collections?

**Problem**: Data isolation and query performance

**Solution**: Separate collection per user (user_{user_id}_documents)

**Benefits**:
- Strong data isolation (no cross-user leakage)
- Faster queries (smaller search space)
- Easy cleanup (delete entire collection)

---

## 16. Future Enhancements

### 16.1 Short-Term (1-3 months)
- [ ] GPU support for Wav2Vec2 inference
- [ ] Model quantization for faster inference
- [ ] Async transcription (background processing)
- [ ] Brute force protection (rate limiting)
- [ ] Data encryption at rest

### 16.2 Medium-Term (3-6 months)
- [ ] Multi-modal emotion recognition (audio + text)
- [ ] Real-time emotion tracking (streaming audio)
- [ ] Advanced RAG (multi-hop reasoning)
- [ ] A/B testing framework for alpha formulas
- [ ] Federated learning for privacy

### 16.3 Long-Term (6-12 months)
- [ ] Mobile SDK (iOS/Android)
- [ ] Edge deployment (on-device inference)
- [ ] Multi-language support
- [ ] Emotion trend analysis
- [ ] Social features (emotion sharing)

---

## Conclusion

This architecture provides a **production-ready, scalable, and maintainable** system for personalized emotion recognition with the following key characteristics:

1. **Layered Architecture**: Clear separation of concerns (API → Services → Data)
2. **Dual-Head ML Pipeline**: Adaptive blending of global and personalized models
3. **Async Processing**: Background training, non-blocking I/O
4. **Data Isolation**: Per-user collections, secure authentication
5. **Performance Optimization**: Caching, lazy loading, connection pooling
6. **Observability**: Comprehensive logging, health checks, metrics
7. **Scalability**: Horizontal scaling, stateless design, shared storage

The system is designed to handle **thousands of concurrent users** with **sub-5-second latency** for audio processing and **sub-10-second training** for personalized models.

---

**Document Version**: 1.0  
**Last Updated**: March 6, 2026  
**Maintained By**: System Architecture Team  
**Contact**: architecture@emotion-api.com
