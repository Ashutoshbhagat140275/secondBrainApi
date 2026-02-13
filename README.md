# RAG Backend with Audio Emotion Analysis

A multi-tenant RAG (Retrieval-Augmented Generation) backend system that processes audio files, analyzes emotions using MFCC features, converts speech to text, and provides intelligent querying capabilities through vector search and local LLM integration.

## Features

- **JWT Authentication**: Secure user registration and login
- **Audio Processing**: Upload and process audio files (WAV, MP3, M4A, FLAC)
- **Emotion Analysis**: Neural embedding-based emotion classification using Wav2Vec2 (happy, sad, angry, neutral, etc.)
- **Speech-to-Text**: Automatic transcription using Whisper
- **Vector Storage**: User-specific vector collections in Qdrant
- **RAG Queries**: Natural language queries with context-aware responses using Ollama
- **Multi-Tenancy**: Complete data isolation per user
- **Dashboard APIs**: Emotion analysis data and user statistics

## Tech Stack

- **Backend**: FastAPI (Python 3.10+)
- **Vector DB**: Qdrant (self-hosted)
- **User DB**: MongoDB
- **Auth**: JWT (python-jose, passlib)
- **Audio Processing**: Wav2Vec2 (neural embeddings), Whisper (speech-to-text)
- **ML**: PyTorch (emotion classifier), HuggingFace Transformers (Wav2Vec2)
- **LLM**: Ollama (local models)
- **Embeddings**: sentence-transformers

## Prerequisites

1. **Python 3.10+**
2. **MongoDB** (running on localhost:27017 or configure in .env)
3. **Qdrant** (running on localhost:6333 or configure in .env)
4. **Ollama** (running on localhost:11434 with a model installed, e.g., `ollama pull llama2`)

## Installation

1. **Clone the repository**:
   ```bash
   cd project1
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Start MongoDB** (if not already running):
   ```bash
   # On Linux/Mac
   mongod
   
   # On Windows, start MongoDB service or run mongod.exe
   ```

6. **Start Qdrant** (using Docker):
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```

7. **Set up Ollama**:
   ```bash
   # Install Ollama from https://ollama.ai
   # Pull a model
   ollama pull llama2
   # Or use mistral, codellama, etc.
   ```

## Configuration

Edit `.env` file with your settings:

```env
# MongoDB
MONGODB_URL=mongodb://localhost:27017
MONGODB_DB_NAME=rag_audio_db

# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=

# JWT
JWT_SECRET_KEY=your-secret-key-change-in-production
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2

# Embedding Model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Audio Processing
AUDIO_UPLOAD_DIR=./uploads
MAX_AUDIO_SIZE_MB=50
```

## Running the Application

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- **API**: http://localhost:8000
- **Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## API Endpoints

### Authentication

- **POST /api/auth/register** - Register a new user
  ```json
  {
    "email": "user@example.com",
    "password": "password123"
  }
  ```

- **POST /api/auth/login** - Login and get JWT token
  ```json
  {
    "email": "user@example.com",
    "password": "password123"
  }
  ```

### Audio Processing

- **POST /api/audio/upload** - Upload and process audio file
  - Headers: `Authorization: Bearer {token}`
  - Body: Multipart form data with audio file

### RAG Queries

- **POST /api/rag/query** - Query the RAG system
  - Headers: `Authorization: Bearer {token}`
  ```json
  {
    "query": "What did I say about the project?",
    "top_k": 5
  }
  ```

### Dashboard

- **GET /api/dashboard/emotions/{user_id}** - Get emotion analysis data
  - Headers: `Authorization: Bearer {token}`
  - Query params: `start_date?`, `end_date?`, `limit?`

- **GET /api/dashboard/stats/{user_id}** - Get user statistics
  - Headers: `Authorization: Bearer {token}`

## Usage Example

1. **Register a user**:
   ```bash
   curl -X POST http://localhost:8000/api/auth/register \
     -H "Content-Type: application/json" \
     -d '{"email": "test@example.com", "password": "test123"}'
   ```

2. **Login**:
   ```bash
   curl -X POST http://localhost:8000/api/auth/login \
     -H "Content-Type: application/json" \
     -d '{"email": "test@example.com", "password": "test123"}'
   ```
   Save the `access_token` from the response.

3. **Upload audio**:
   ```bash
   curl -X POST http://localhost:8000/api/audio/upload \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -F "file=@audio.wav"
   ```

4. **Query RAG**:
   ```bash
   curl -X POST http://localhost:8000/api/rag/query \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"query": "What emotions did I express?", "top_k": 5}'
   ```

## Project Structure

```
project1/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app entry
│   ├── config.py               # Configuration
│   ├── models/                 # Database models
│   ├── schemas/                # Pydantic schemas
│   ├── services/               # Business logic
│   ├── routers/                # API endpoints
│   ├── db/                     # Database connections
│   └── middleware/             # Auth middleware
├── requirements.txt
├── requirements.md             # Requirements document
├── design.md                   # Design document
└── README.md
```

## Emotion Analysis Model

The system uses a neural embedding-based approach with Wav2Vec2:

1. **Feature Extraction**: Wav2Vec2 pretrained model extracts 768-dimensional neural embeddings from audio
2. **Classification**: A trained PyTorch classifier head maps embeddings to 8 emotion classes
3. **Training**: Use `python -m training.train_wav2vec2` to train on RAVDESS and CREMA-D datasets

Key benefits over manual feature extraction:
- 3-5x faster processing (3-5 seconds vs 15 seconds for 20-second audio)
- More robust to noise, silence, and unvoiced segments
- No pitch extraction failures or NaN issues
- Captures semantic patterns like prosody, rhythm, and speaking style

## Alpha Engine: Dual-Head Blending Strategy

The system uses a **dual-head architecture** that combines predictions from two models:
- **Global Head**: Trained on all users' data (general emotion patterns)
- **User Head**: Trained on individual user's feedback (personalized patterns)

The **Alpha Engine** determines how to blend these predictions using a dynamic weight (alpha):
```
Final Prediction = alpha × Global Prediction + (1 - alpha) × User Prediction
```

### Sigmoid vs Linear Formulas

The system supports two blending strategies, controlled by the `USE_SIGMOID_ALPHA` flag in `feature_config.py`.

#### Linear Formula (Legacy)

**Formula:**
```
α = 0.5 + 0.3·C_g - 0.2·min(feedback_count/100, 1.0)
Clamped to [0.3, 1.0]
```

**Characteristics:**
- Simple additive combination of confidence and feedback
- Hard-coded thresholds and clamping
- Special case: Returns 1.0 for feedback_count < 20
- Linear decay with feedback count

**When to use:** Stable baseline, well-tested in production

#### Sigmoid Formula (Recommended)

**Formula:**
```
alpha_data = 1 / (1 + N/K)
alpha_conf = 1 / (1 + exp(-β(C_g - τ)))
alpha = alpha_data × alpha_conf
```

**Characteristics:**
- **Separation of concerns**: Data availability (feedback) and confidence are independent
- **Smooth transitions**: Exponential decay for feedback, S-curve for confidence
- **Multiplicative logic**: Both components must agree to trust global head
- **No hard clamping**: Natural bounds from sigmoid function (0, 1)
- **Tunable**: Three hyperparameters allow fine-grained control

**When to use:** Better personalization, smoother behavior, more intuitive tuning

**Key Differences:**

| Aspect | Linear | Sigmoid |
|--------|--------|---------|
| Feedback decay | Linear | Exponential |
| Confidence response | Linear | S-curve (sigmoid) |
| Combination | Additive | Multiplicative |
| Bounds | Hard clamp [0.3, 1.0] | Natural [0, 1] |
| New user behavior | Always 1.0 (N < 20) | Responds to confidence |
| Tuning | Fixed coefficients | Three hyperparameters |

### Hyperparameters

The sigmoid formula uses three tunable hyperparameters defined in `app/services/feature_config.py`:

#### K (Feedback Scale Constant)

**Default:** 50

**Effect:** Controls how quickly alpha_data decays as users provide feedback

**Behavior:**
- At N = 0: alpha_data = 1.0 (no feedback → trust global fully)
- At N = K: alpha_data = 0.5 (equal weight between global and user)
- At N = 2K: alpha_data = 0.33 (favor user head)
- As N → ∞: alpha_data → 0.0 (trust user head only)

**Tuning guide:**
- **K = 25**: Fast personalization (reaches 50/50 blend at 25 feedback samples)
- **K = 50**: Medium personalization (default, balanced approach)
- **K = 100**: Slow personalization (requires more feedback before trusting user head)

**When to adjust:**
- Increase K if users complain about premature personalization
- Decrease K if users want faster adaptation to their preferences

#### τ (Tau - Confidence Threshold)

**Default:** 0.6

**Effect:** Sets the confidence level at which alpha_conf = 0.5 (equal weight)

**Behavior:**
- C_g < τ: alpha_conf < 0.5 (low confidence → favor user head)
- C_g = τ: alpha_conf = 0.5 (threshold → equal weight)
- C_g > τ: alpha_conf > 0.5 (high confidence → trust global head)

**Tuning guide:**
- **τ = 0.5**: Lower threshold (trust global head more easily)
- **τ = 0.6**: Medium threshold (default, balanced)
- **τ = 0.7**: Higher threshold (require high confidence to trust global)

**When to adjust:**
- Decrease τ if global head is well-calibrated and reliable
- Increase τ if global head tends to be overconfident

#### β (Beta - Sigmoid Sharpness)

**Default:** 10

**Effect:** Controls how steep the sigmoid transition is around τ

**Behavior:**
- Low β: Gentle, gradual transition (wide confidence range affects alpha)
- High β: Sharp, decisive transition (narrow confidence range affects alpha)

**Tuning guide:**
- **β = 5**: Gentle transition (smooth blending across confidence range)
- **β = 10**: Medium transition (default, balanced)
- **β = 20**: Sharp transition (decisive switching between heads)

**When to adjust:**
- Decrease β for smoother, more gradual blending
- Increase β for more decisive switching based on confidence

### Example Alpha Values

Here are concrete examples showing how alpha behaves under different scenarios:

#### Scenario 1: New User (N = 0 feedback samples)

**Linear formula:**
```
alpha = 1.0 (hardcoded for N < 20)
```

**Sigmoid formula:**
```
alpha_data = 1.0 (no feedback)

Low confidence (C_g = 0.5):
  alpha_conf = 0.27 → alpha = 0.27 (favor user head despite no feedback)

Medium confidence (C_g = 0.7):
  alpha_conf = 0.73 → alpha = 0.73 (trust global head)

High confidence (C_g = 0.9):
  alpha_conf = 0.95 → alpha = 0.95 (strongly trust global head)
```

**Insight:** Sigmoid responds to confidence even for new users, while linear always trusts global.

#### Scenario 2: Medium Feedback (N = 50 samples)

**Linear formula:**
```
C_g = 0.5 → alpha = 0.5 + 0.15 - 0.1 = 0.55
C_g = 0.7 → alpha = 0.5 + 0.21 - 0.1 = 0.61
C_g = 0.9 → alpha = 0.5 + 0.27 - 0.1 = 0.67
```

**Sigmoid formula:**
```
alpha_data = 0.5 (at threshold K = 50)

C_g = 0.5 → alpha_conf = 0.27 → alpha = 0.14 (strongly favor user)
C_g = 0.7 → alpha_conf = 0.73 → alpha = 0.37 (favor user)
C_g = 0.9 → alpha_conf = 0.95 → alpha = 0.48 (nearly equal)
```

**Insight:** At medium feedback, sigmoid favors user head more aggressively than linear.

#### Scenario 3: High Feedback (N = 100 samples)

**Linear formula:**
```
C_g = 0.5 → alpha = 0.5 + 0.15 - 0.2 = 0.45
C_g = 0.7 → alpha = 0.5 + 0.21 - 0.2 = 0.51
C_g = 0.9 → alpha = 0.5 + 0.27 - 0.2 = 0.57
```

**Sigmoid formula:**
```
alpha_data = 0.33 (lots of feedback)

C_g = 0.5 → alpha_conf = 0.27 → alpha = 0.09 (strongly favor user)
C_g = 0.7 → alpha_conf = 0.73 → alpha = 0.24 (favor user)
C_g = 0.9 → alpha_conf = 0.95 → alpha = 0.31 (still favor user)
```

**Insight:** With lots of feedback, sigmoid strongly favors user head regardless of confidence.

#### Scenario 4: Edge Case - Very High Confidence, No Feedback

**Linear formula:**
```
C_g = 0.95, N = 0 → alpha = 1.0 (hardcoded)
```

**Sigmoid formula:**
```
C_g = 0.95, N = 0:
  alpha_data = 1.0
  alpha_conf = 0.98
  alpha = 0.98 (trust global head)
```

**Insight:** Both formulas trust global head, but sigmoid is slightly more conservative.

### Configuration and Deployment

#### Enabling Sigmoid Formula

Edit `app/services/feature_config.py`:

```python
# Alpha Engine Configuration
USE_SIGMOID_ALPHA = True  # Enable sigmoid formula
ALPHA_FEEDBACK_SCALE_K = 50
ALPHA_CONFIDENCE_THRESHOLD_TAU = 0.6
ALPHA_SIGMOID_SHARPNESS_BETA = 10
```

#### Monitoring Alpha Values

The API response includes alpha components for debugging:

```json
{
  "emotion": "happy",
  "confidence": 0.85,
  "blend_weight": 0.42,
  "alpha_data": 0.67,
  "alpha_conf": 0.63,
  "alpha_formula": "sigmoid",
  "global_emotion": "happy",
  "global_confidence": 0.72,
  "user_emotion": "excited",
  "user_confidence": 0.91
}
```

#### Tuning Workflow

1. **Start with defaults:** K=50, τ=0.6, β=10
2. **Monitor alpha distribution:** Check mean, std, percentiles in logs
3. **Adjust K** if personalization speed is wrong:
   - Users complain about slow adaptation → decrease K
   - Users complain about erratic predictions → increase K
4. **Adjust τ** if confidence calibration is off:
   - Global head is reliable but underused → decrease τ
   - Global head is overconfident → increase τ
5. **Adjust β** if transitions are too abrupt or too gradual:
   - Want smoother blending → decrease β
   - Want more decisive switching → increase β

#### Rollback Plan

If issues arise with sigmoid formula:

1. Set `USE_SIGMOID_ALPHA = False` in `feature_config.py`
2. Restart the service
3. System automatically reverts to linear formula
4. No data loss or API changes required

### Performance Considerations

- Alpha computation adds < 1ms overhead per prediction
- Sigmoid uses numpy's optimized exp() function
- All three alpha components are logged at DEBUG level
- Production deployments should monitor alpha distributions for anomalies

## Development

### Running Tests

```bash
# Add tests in tests/ directory
pytest
```

### Code Style

```bash
black app/
flake8 app/
```

## Production Deployment

1. Set `DEBUG=False` in `.env`
2. Use a strong `JWT_SECRET_KEY`
3. Configure CORS appropriately
4. Use a production ASGI server (e.g., Gunicorn with Uvicorn workers)
5. Set up proper logging and monitoring
6. Use cloud storage for audio files (S3, etc.)
7. Set up database backups

## Troubleshooting

### MongoDB Connection Issues
- Ensure MongoDB is running: `mongosh` or check service status
- Verify connection string in `.env`

### Qdrant Connection Issues
- Check if Qdrant is running: `curl http://localhost:6333/collections`
- Verify Qdrant URL in `.env`

### Ollama Issues
- Ensure Ollama is running: `ollama list`
- Pull the model: `ollama pull llama2`
- Check Ollama URL in `.env`

### Audio Processing Errors
- Ensure audio file format is supported (WAV, MP3, M4A, FLAC)
- Check file size (default max: 50MB)
- Verify librosa and soundfile are installed correctly

## License

This project is provided as-is for educational and development purposes.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Support

For issues and questions, please open an issue on the repository.

