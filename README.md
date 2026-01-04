# RAG Backend with Audio Emotion Analysis

A multi-tenant RAG (Retrieval-Augmented Generation) backend system that processes audio files, analyzes emotions using MFCC features, converts speech to text, and provides intelligent querying capabilities through vector search and local LLM integration.

## Features

- **JWT Authentication**: Secure user registration and login
- **Audio Processing**: Upload and process audio files (WAV, MP3, M4A, FLAC)
- **Emotion Analysis**: MFCC-based emotion classification (happy, sad, angry, neutral, etc.)
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
- **Audio Processing**: librosa (MFCC), Whisper (speech-to-text)
- **ML**: scikit-learn (emotion model from MFCC features)
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

The current implementation uses a placeholder emotion classification model. To use a trained model:

1. Train a model on MFCC features (using scikit-learn, TensorFlow, etc.)
2. Save the model to a file
3. Update `app/services/emotion_analyzer.py` to load and use your trained model

Example datasets for training:
- RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
- CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset)

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

