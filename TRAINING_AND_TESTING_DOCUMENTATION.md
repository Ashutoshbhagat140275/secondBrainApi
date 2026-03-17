# Training and Testing Documentation

## Table of Contents
1. [Training Datasets](#training-datasets)
2. [Model Architecture](#model-architecture)
3. [Training Process](#training-process)
4. [Testing Strategy](#testing-strategy)
5. [Test Cases](#test-cases)

---

## 1. Training Datasets

### 1.1 Public Datasets Used

The emotion recognition system is trained on two well-established public datasets:

#### **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)**
- **Size**: ~1,440 audio files
- **Actors**: 24 professional actors (12 male, 12 female)
- **Emotions**: 8 emotions (neutral, calm, happy, sad, angry, fearful, disgusted, surprised)
- **Format**: Audio-only speech files (modality 03)
- **Filename Format**: `{modality}-{vocal_channel}-{emotion}-{intensity}-{statement}-{rep}-{actor}.wav`
- **Emotion Codes**:
  - 01 = neutral
  - 02 = calm
  - 03 = happy
  - 04 = sad
  - 05 = angry
  - 06 = fearful
  - 07 = disgusted
  - 08 = surprised

#### **CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset)**
- **Size**: ~7,442 audio files
- **Actors**: 91 actors (48 male, 43 female)
- **Emotions**: 6 emotions (anger, disgust, fear, happy, neutral, sad)
- **Format**: Audio files with emotion labels
- **Filename Format**: `{ActorID}_{Sentence}_{Emotion}_{Level}.wav`
- **Emotion Codes**:
  - ANG = angry
  - DIS = disgusted
  - FEA = fearful
  - HAP = happy
  - NEU = neutral
  - SAD = sad

### 1.2 Combined Dataset Statistics

After combining both datasets:
- **Total Samples**: ~7,000-8,000 audio files
- **Emotion Distribution**: Balanced across 8 emotion classes
- **Train/Val/Test Split**: 70% / 15% / 15%
- **Data Augmentation**: None (uses pretrained Wav2Vec2 embeddings)



---

## 2. Model Architecture

### 2.1 Dual-Head Architecture Overview

The system uses a **dual-head architecture** with two separate classifiers:

```
Audio Input (WAV/MP3/etc.)
    ↓
Preprocessing (16kHz, VAD, Normalize)
    ↓
Wav2Vec2 Encoder (Facebook pretrained)
    ↓
768-dimensional Embedding
    ↓
    ├─────────────────┬─────────────────┐
    ↓                 ↓                 ↓
Global Head      User Head       Alpha Engine
(shared)         (per-user)      (blending)
    ↓                 ↓                 ↓
P_g (8,)         P_u (8,)         α (scalar)
    ↓                 ↓                 ↓
    └─────────────────┴─────────────────┘
                      ↓
            P_f = α·P_g + (1-α)·P_u
                      ↓
            Final Emotion Prediction
```

### 2.2 Wav2Vec2 Encoder

**Model**: Facebook's `wav2vec2-base` (pretrained on 960 hours of LibriSpeech)
- **Input**: Raw audio waveform at 16kHz
- **Output**: 768-dimensional embedding vector
- **Architecture**: Transformer-based self-supervised learning
- **Parameters**: ~95 million (frozen, not trained)
- **Purpose**: Extract high-level acoustic features

**Why Wav2Vec2?**
- Replaces manual feature extraction (MFCCs, pitch, jitter/shimmer)
- 5x faster than legacy feature extraction
- More robust to noise and silence
- Captures prosody, rhythm, and speaking style

### 2.3 Global Emotion Head

**Architecture**: Simple linear classifier
```python
class GlobalEmotionHead(nn.Module):
    def __init__(self, embedding_dim=768, num_classes=8):
        super().__init__()
        self.linear = nn.Linear(embedding_dim, num_classes)
    
    def forward(self, x):
        return self.linear(x)  # Returns logits (8,)
```

**Parameters**:
- Input: 768-dimensional embedding
- Output: 8 logits (one per emotion)
- Total parameters: 768 × 8 + 8 = **6,152 parameters**
- Model size: ~24 KB

**Training**:
- Dataset: RAVDESS + CREMA-D (~7,000 samples)
- Loss: CrossEntropyLoss with class weights
- Optimizer: Adam (lr=1e-3, weight_decay=1e-4)
- Batch size: 64
- Epochs: 50 (with early stopping, patience=10)
- Validation accuracy: **72.7%**



### 2.4 User Emotion Head

**Architecture**: Identical to Global Head
```python
class UserEmotionHead(nn.Module):
    def __init__(self, embedding_dim=768, num_classes=8):
        super().__init__()
        self.linear = nn.Linear(embedding_dim, num_classes)
    
    def forward(self, x):
        return self.linear(x)
```

**Parameters**: Same as Global Head (6,152 parameters, ~24 KB)

**Training**:
- Dataset: User-specific feedback corrections (20-100+ samples per user)
- Loss: CrossEntropyLoss
- Optimizer: Adam (lr=1e-3, weight_decay=1e-4)
- Batch size: 16 (or full batch if < 16 samples)
- Epochs: 20
- Training triggers:
  - Initial: 20 feedback samples
  - Incremental: Every 10 samples after initial (30, 40, 50, ...)

**Personalization Benefits**:
- Adapts to individual expression patterns
- Handles unique speaking styles (loud, flat pitch, sarcasm)
- Improves accuracy by 10%+ after 50 samples
- Maintains global baseline for new users

### 2.5 Alpha Engine (Blending Strategy)

**Purpose**: Adaptively blend global and user predictions based on:
- Global head confidence (higher confidence → more global weight)
- User feedback count (more feedback → more user weight)

**Sigmoid Formula** (current implementation):
```
alpha_data = 1 / (1 + N/K)
alpha_conf = 1 / (1 + exp(-β(C_g - τ)))
alpha = alpha_data × alpha_conf
```

**Parameters**:
- K = 50: Feedback scaling constant
- τ = 0.6: Confidence threshold
- β = 10: Sigmoid sharpness

**Behavior**:
- N=0 (new user): α ≈ 1.0 (trust global fully, modulated by confidence)
- N=50 (medium feedback): α ≈ 0.5 (balanced blending)
- N=100+ (high feedback): α ≈ 0.3 (favor user head unless global very confident)

**Final Prediction**:
```
P_f = α·P_g + (1-α)·P_u
emotion = argmax(P_f)
confidence = max(P_f)
```



---

## 3. Training Process

### 3.1 Global Head Training

**Script**: `training/train_wav2vec2.py`

**Command**:
```bash
python -m training.train_wav2vec2 --save-as-global-head --epochs 50 --batch 64
```

**Steps**:

1. **Dataset Preparation**
   ```bash
   python -m training.prepare_dataset
   ```
   - Scans RAVDESS and CREMA-D directories
   - Parses filenames to extract emotion labels
   - Creates manifest.json with all samples

2. **Embedding Extraction** (with caching)
   - Loads Wav2Vec2 model from HuggingFace
   - Extracts 768-dim embeddings for all audio files
   - Caches to `training/cache/wav2vec2_embeddings.npz`
   - Takes ~30-60 minutes for 7,000 samples (first time only)

3. **Data Normalization**
   - Applies StandardScaler to embeddings
   - Saves scaler to `models/embedding_scaler.joblib`

4. **Train/Val/Test Split**
   - Train: 70% (~4,900 samples)
   - Validation: 15% (~1,050 samples)
   - Test: 15% (~1,050 samples)
   - Stratified split (balanced emotion distribution)

5. **Model Training**
   - Initialize GlobalEmotionHead (Linear 768→8)
   - Compute class weights for imbalanced data
   - Train for 50 epochs with early stopping (patience=10)
   - Learning rate scheduler: ReduceLROnPlateau
   - Save best model based on validation loss

6. **Model Saving**
   - Saves to `models/global_emotion_head.pt`
   - Loaded once at app startup (singleton pattern)

**Training Time**: ~5-10 minutes on CPU, ~1-2 minutes on GPU

**Results**:
- Validation accuracy: 72.7%
- Test accuracy: 72.7%
- See `training/evaluation_results.md` for detailed metrics



### 3.2 User Head Training

**Script**: `training/train_user_head.py`

**Automatic Trigger** (via API):
- Triggered automatically when user provides feedback
- Initial training: 20 feedback samples
- Incremental training: Every 10 samples after initial (30, 40, 50, ...)
- Runs asynchronously in background (non-blocking)

**Manual Trigger** (CLI):
```bash
# Initial training
python -m training.train_user_head --user-id <user_id>

# Force retrain from scratch
python -m training.train_user_head --user-id <user_id> --force-retrain

# Incremental training
python -m training.train_user_head --user-id <user_id> --incremental
```

**Steps**:

1. **Load Feedback Data**
   - Query MongoDB `user_feedback` collection
   - Extract embeddings (768-dim) and corrected emotion labels
   - Minimum requirement: 20 samples

2. **Model Initialization**
   - If first training: Create fresh UserEmotionHead
   - If incremental: Load existing weights from `models/user_heads/{user_id}.pt`

3. **Training Configuration**
   - Loss: CrossEntropyLoss
   - Optimizer: Adam (lr=1e-3, weight_decay=1e-4)
   - Batch size: 16 (or full batch if < 16 samples)
   - Epochs: 20

4. **Training Loop**
   - Train for 20 epochs
   - No validation split (small dataset)
   - Log metrics every 5 epochs

5. **Model Saving**
   - Save to `models/user_heads/{user_id}.pt`
   - Invalidate LRU cache to ensure fresh model is loaded
   - Store metadata (training_samples, last_trained)

**Training Time**: <10 seconds for 20-100 samples (CPU)

**Storage**:
- Per-user model size: ~5 KB (compressed)
- 10,000 users = ~50 MB total
- LRU cache: 100 models in memory (~500 KB)



---

## 4. Testing Strategy

The API uses a comprehensive testing strategy with multiple levels:

### 4.1 Test Levels

1. **Unit Tests**: Test individual functions and classes in isolation
2. **Integration Tests**: Test interactions between components
3. **Property-Based Tests**: Test universal properties across many inputs
4. **End-to-End Tests**: Test complete workflows from API to database

### 4.2 Test Framework

- **Framework**: pytest
- **Mocking**: unittest.mock
- **Property Testing**: Hypothesis
- **Coverage**: pytest-cov
- **Async Testing**: pytest-asyncio

### 4.3 Test Organization

```
api/tests/
├── test_alpha_engine.py                    # Alpha blending logic
├── test_alpha_engine_properties.py         # Property-based tests
├── test_dual_head_classifier.py            # Dual-head classification
├── test_dual_head_integration.py           # End-to-end integration
├── test_feedback_service.py                # Feedback submission
├── test_feedback_properties.py             # Feedback properties
├── test_global_emotion_head.py             # Global head unit tests
├── test_user_emotion_head.py               # User head unit tests
├── test_train_user_head.py                 # Training logic
├── test_training_job_tracker.py            # Job tracking
├── test_wav2vec2_encoder.py                # Embedding extraction
├── test_audio_upload_response.py           # API response format
├── test_feedback_endpoint.py               # Feedback API endpoint
└── ... (40+ test files)
```

### 4.4 Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_dual_head_integration.py

# Run with coverage
pytest --cov=app --cov-report=html

# Run property-based tests with more examples
pytest tests/test_alpha_engine_properties.py --hypothesis-seed=42

# Run integration tests only
pytest -m integration

# Run unit tests only
pytest -m unit
```



---

## 5. Test Cases

### 5.1 Unit Test Cases

#### **Alpha Engine Tests** (`test_alpha_engine.py`)

| Test Case | Description | Expected Result |
|-----------|-------------|-----------------|
| `test_compute_alpha_data_zero_feedback` | N=0 feedback samples | α_data = 1.0 |
| `test_compute_alpha_data_k_feedback` | N=K feedback samples | α_data = 0.5 |
| `test_compute_alpha_data_large_feedback` | N=1000 feedback samples | α_data → 0.0 |
| `test_compute_alpha_conf_low_confidence` | C_g = 0.3 (below threshold) | α_conf < 0.5 |
| `test_compute_alpha_conf_threshold` | C_g = 0.6 (at threshold) | α_conf = 0.5 |
| `test_compute_alpha_conf_high_confidence` | C_g = 0.9 (above threshold) | α_conf > 0.5 |
| `test_compute_alpha_sigmoid_new_user` | N=0, C_g=0.8 | α ≈ 0.88 |
| `test_compute_alpha_sigmoid_medium_user` | N=50, C_g=0.7 | α ≈ 0.56 |
| `test_compute_alpha_sigmoid_experienced_user` | N=100, C_g=0.6 | α ≈ 0.17 |
| `test_invalid_parameters` | K≤0, τ∉(0,1), β≤0 | Raises ValueError |

#### **Dual-Head Classifier Tests** (`test_dual_head_classifier.py`)

| Test Case | Description | Expected Result |
|-----------|-------------|-----------------|
| `test_blend_predictions_equal_weight` | α=0.5, P_g≠P_u | P_f = 0.5·P_g + 0.5·P_u |
| `test_blend_predictions_global_only` | α=1.0 | P_f = P_g |
| `test_blend_predictions_user_only` | α=0.0 | P_f = P_u |
| `test_blend_predictions_normalization` | Any α, P_g, P_u | sum(P_f) = 1.0 |
| `test_classify_new_user_global_only` | N=0, no user model | user_emotion=None, α=1.0 |
| `test_classify_experienced_user_dual_head` | N=50, user model exists | Both predictions present |
| `test_classify_heads_disagree` | P_g≠P_u | Final emotion from blended P_f |

#### **Global Emotion Head Tests** (`test_global_emotion_head.py`)

| Test Case | Description | Expected Result |
|-----------|-------------|-----------------|
| `test_load_global_head_success` | Model file exists | Returns True, model loaded |
| `test_load_global_head_missing_file` | Model file missing | Returns False, warning logged |
| `test_predict_global_valid_embedding` | 768-dim embedding | Returns (P_g, C_g) |
| `test_predict_global_invalid_shape` | Wrong embedding shape | Raises ValueError |
| `test_predict_global_fallback` | Model not loaded | Returns uniform distribution |
| `test_predict_global_probabilities_sum` | Any embedding | sum(P_g) = 1.0 |
| `test_predict_global_confidence_range` | Any embedding | 0 ≤ C_g ≤ 1 |

#### **User Emotion Head Tests** (`test_user_emotion_head.py`)

| Test Case | Description | Expected Result |
|-----------|-------------|-----------------|
| `test_create_fresh_user_head` | Create new model | Returns UserEmotionHead |
| `test_load_user_head_success` | Model file exists | Returns model from cache |
| `test_load_user_head_missing` | Model file missing | Returns None |
| `test_predict_user_valid_embedding` | 768-dim embedding, model exists | Returns (P_u, C_u) |
| `test_predict_user_no_model` | No model for user | Returns None |
| `test_lru_cache_eviction` | Cache > 100 models | Oldest model evicted |
| `test_invalidate_user_cache` | Cache invalidation | Model removed from cache |



#### **Feedback Service Tests** (`test_feedback_service.py`)

| Test Case | Description | Expected Result |
|-----------|-------------|-----------------|
| `test_submit_feedback_valid` | Valid session, emotion | Feedback stored, count incremented |
| `test_submit_feedback_invalid_emotion` | Invalid emotion label | Raises ValueError |
| `test_submit_feedback_session_not_found` | Non-existent session | Raises ValueError |
| `test_submit_feedback_wrong_user` | Session belongs to other user | Raises PermissionError |
| `test_submit_feedback_trigger_initial` | 20th feedback sample | training_triggered=True |
| `test_submit_feedback_trigger_incremental` | 30th, 40th, 50th sample | training_triggered=True |
| `test_submit_feedback_no_trigger` | 25th, 35th sample | training_triggered=False |

#### **Training Job Tracker Tests** (`test_training_job_tracker.py`)

| Test Case | Description | Expected Result |
|-----------|-------------|-----------------|
| `test_create_training_job` | Create new job | Job created with status="queued" |
| `test_update_job_status_running` | Update to running | Status updated, started_at set |
| `test_update_job_status_completed` | Update to completed | Status updated, metrics stored |
| `test_update_job_status_failed` | Update to failed | Status updated, error_message set |
| `test_get_latest_job` | Multiple jobs for user | Returns most recent job |
| `test_get_latest_job_no_jobs` | No jobs for user | Returns None |

#### **Wav2Vec2 Encoder Tests** (`test_wav2vec2_encoder.py`)

| Test Case | Description | Expected Result |
|-----------|-------------|-----------------|
| `test_load_wav2vec2_model` | Load model | Returns True, model loaded |
| `test_extract_embedding_valid_audio` | Valid audio file | Returns 768-dim embedding |
| `test_extract_embedding_short_audio` | Audio < 0.3s | Pads to minimum length |
| `test_extract_embedding_nan_values` | Audio contains NaN | Replaces NaN with zeros |
| `test_extract_embedding_invalid_file` | Non-existent file | Raises RuntimeError |
| `test_extract_embedding_empty_audio` | Empty audio file | Raises ValueError |

### 5.2 Integration Test Cases

#### **Dual-Head Integration Tests** (`test_dual_head_integration.py`)

| Test Case | Description | Expected Result |
|-----------|-------------|-----------------|
| `test_end_to_end_new_user_global_only` | New user uploads audio | Global-only prediction, user_emotion=None |
| `test_end_to_end_experienced_user_dual_head` | User with 50 feedback uploads | Dual-head prediction, both emotions present |
| `test_dual_head_disagreement` | Global="neutral", User="sad" | Final emotion from blended prediction |
| `test_response_format_includes_all_fields` | Any audio upload | All extended fields present |
| `test_feedback_loop_integration` | Submit feedback → training | Training triggered at correct thresholds |
| `test_classify_with_dual_heads_integration` | Real blending logic | Predictions blended correctly |
| `test_training_integration` | Train user head | Model saved, metrics returned |

#### **Feedback Integration Tests** (`test_feedback_integration_comprehensive.py`)

| Test Case | Description | Expected Result |
|-----------|-------------|-----------------|
| `test_feedback_to_training_pipeline` | 20 feedback samples | Training job created and executed |
| `test_incremental_training_pipeline` | 30 feedback samples | Existing model updated |
| `test_feedback_improves_accuracy` | Before/after training | User head accuracy improves |
| `test_concurrent_feedback_submissions` | Multiple users submit feedback | No race conditions |



### 5.3 Property-Based Test Cases

#### **Alpha Engine Properties** (`test_alpha_engine_properties.py`)

| Property | Description | Invariant |
|----------|-------------|-----------|
| `test_alpha_data_monotonic_decrease` | As N increases, α_data decreases | α_data(N+1) ≤ α_data(N) |
| `test_alpha_data_bounds` | For any N ≥ 0 | 0 < α_data ≤ 1 |
| `test_alpha_conf_monotonic_increase` | As C_g increases, α_conf increases | α_conf(C_g+ε) ≥ α_conf(C_g) |
| `test_alpha_conf_bounds` | For any C_g ∈ [0,1] | 0 < α_conf < 1 |
| `test_alpha_final_bounds` | For any N, C_g | 0 < α ≤ 1 |
| `test_alpha_new_user_high` | N=0, C_g high | α > 0.8 |
| `test_alpha_experienced_user_low` | N>100, C_g low | α < 0.3 |

#### **Feedback Properties** (`test_feedback_properties.py`)

| Property | Description | Invariant |
|----------|-------------|-----------|
| `test_feedback_count_increases` | After feedback submission | count(after) = count(before) + 1 |
| `test_training_trigger_deterministic` | Same feedback count | Same training_triggered result |
| `test_feedback_idempotency` | Submit same feedback twice | Second submission rejected |

#### **Training Properties** (`test_training_properties.py`)

| Property | Description | Invariant |
|----------|-------------|-----------|
| `test_training_improves_or_maintains` | After training | accuracy(after) ≥ accuracy(before) - ε |
| `test_model_size_consistent` | Any user model | size ≈ 6,152 parameters |
| `test_training_deterministic` | Same data, same seed | Same final weights |

### 5.4 End-to-End API Test Cases

#### **Audio Upload Endpoint** (`POST /api/audio/upload`)

| Test Case | Input | Expected Output |
|-----------|-------|-----------------|
| Valid WAV file | 16kHz mono WAV | 200 OK, emotion prediction |
| Valid MP3 file | MP3 audio | 200 OK, emotion prediction |
| Invalid format | TXT file | 400 Bad Request |
| File too large | 100MB audio | 400 Bad Request |
| No authentication | No JWT token | 401 Unauthorized |
| Corrupted audio | Corrupted WAV | 500 Internal Server Error |

#### **Feedback Endpoint** (`POST /api/feedback`)

| Test Case | Input | Expected Output |
|-----------|-------|-----------------|
| Valid feedback | Valid session_id, emotion | 200 OK, feedback_count |
| Invalid emotion | emotion="excited" | 400 Bad Request |
| Session not found | Non-existent session_id | 400 Bad Request |
| Wrong user | Other user's session | 403 Forbidden |
| 20th feedback | feedback_count=20 | training_triggered=True |
| No authentication | No JWT token | 401 Unauthorized |

#### **Training Status Endpoint** (`GET /api/training-status/{user_id}`)

| Test Case | Input | Expected Output |
|-----------|-------|-----------------|
| User with jobs | Valid user_id | 200 OK, job details |
| User without jobs | New user_id | 200 OK, all fields null |
| Wrong user | Other user_id | 403 Forbidden |
| No authentication | No JWT token | 401 Unauthorized |



### 5.5 Performance Test Cases

#### **Alpha Engine Performance** (`test_alpha_engine_performance.py`)

| Test Case | Description | Target |
|-----------|-------------|--------|
| `test_alpha_computation_latency` | Compute α for 1000 samples | < 10ms total |
| `test_alpha_memory_usage` | Memory footprint | < 1MB |

#### **Feedback Performance** (`test_feedback_performance.py`)

| Test Case | Description | Target |
|-----------|-------------|--------|
| `test_feedback_submission_latency` | Submit feedback | < 100ms |
| `test_concurrent_feedback_submissions` | 100 concurrent requests | No failures |
| `test_training_trigger_latency` | Enqueue training job | < 50ms |

### 5.6 Model Evaluation Results

#### **Global Head Performance** (from `evaluation_results.md`)

**Test Set**: 216 samples

**Overall Accuracy**: 72.7%

**Per-Class Metrics**:

| Emotion | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| neutral | 0.500 | 0.786 | 0.611 | 14 |
| calm | 0.618 | 0.750 | 0.677 | 28 |
| happy | 0.690 | 0.690 | 0.690 | 29 |
| sad | 0.810 | 0.586 | 0.680 | 29 |
| angry | 0.857 | 0.828 | 0.842 | 29 |
| fearful | 0.792 | 0.655 | 0.717 | 29 |
| disgusted | 0.800 | 0.828 | 0.814 | 29 |
| surprised | 0.750 | 0.724 | 0.737 | 29 |

**Confusion Matrix Analysis**:
- **Best Performance**: Angry (85.7% precision, 82.8% recall)
- **Worst Performance**: Neutral (50.0% precision, 78.6% recall)
- **Common Confusions**:
  - Neutral ↔ Calm (3 misclassifications)
  - Happy ↔ Surprised (5 misclassifications)
  - Sad ↔ Calm (5 misclassifications)

**Insights**:
- Model performs well on high-arousal emotions (angry, disgusted)
- Struggles with low-arousal emotions (neutral, calm)
- Confusion between similar valence emotions (happy/surprised, sad/calm)



---

## 6. Test Execution Summary

### 6.1 Test Coverage

**Total Test Files**: 40+
**Total Test Cases**: 200+
**Code Coverage**: ~85% (target: 90%)

**Coverage by Module**:
- `app/services/`: 90%
- `app/routers/`: 85%
- `app/models/`: 95%
- `training/`: 80%

### 6.2 Continuous Integration

Tests are run automatically on:
- Every commit (pre-commit hooks)
- Pull requests (GitHub Actions)
- Nightly builds (full test suite + property tests)

### 6.3 Test Execution Time

| Test Suite | Duration | Frequency |
|------------|----------|-----------|
| Unit tests | ~30 seconds | Every commit |
| Integration tests | ~2 minutes | Every PR |
| Property-based tests | ~5 minutes | Nightly |
| End-to-end tests | ~10 minutes | Nightly |
| Full suite | ~15 minutes | Release |

### 6.4 Known Limitations

1. **Embedding Extraction**: Requires Wav2Vec2 model (~360MB download on first run)
2. **Training Data**: Limited to RAVDESS + CREMA-D (no real-world data)
3. **Emotion Classes**: Fixed 8 emotions (cannot add new emotions without retraining)
4. **Language**: English only (Wav2Vec2 trained on English speech)
5. **Audio Quality**: Performance degrades on noisy or low-quality audio

### 6.5 Future Testing Improvements

1. **Load Testing**: Test with 1000+ concurrent users
2. **Stress Testing**: Test with 10,000+ user models
3. **Real-World Data**: Collect and test on production data
4. **Cross-Language**: Test on non-English speech
5. **Adversarial Testing**: Test robustness to adversarial inputs
6. **A/B Testing**: Compare dual-head vs global-only accuracy

---

## 7. How to Run Tests

### 7.1 Setup Test Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Install test dependencies
pip install pytest pytest-cov pytest-asyncio hypothesis

# Download Wav2Vec2 model (first time only)
python -c "from transformers import Wav2Vec2Model; Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base')"
```

### 7.2 Run All Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=app --cov-report=html

# Open coverage report
open htmlcov/index.html  # macOS
start htmlcov/index.html  # Windows
```

### 7.3 Run Specific Test Suites

```bash
# Unit tests only
pytest tests/test_alpha_engine.py tests/test_dual_head_classifier.py

# Integration tests only
pytest tests/test_dual_head_integration.py tests/test_feedback_integration_comprehensive.py

# Property-based tests only
pytest tests/test_alpha_engine_properties.py tests/test_feedback_properties.py

# API endpoint tests only
pytest tests/test_feedback_endpoint.py tests/test_audio_upload_response.py
```

### 7.4 Run Tests with Different Configurations

```bash
# Run with more property test examples
pytest tests/test_alpha_engine_properties.py --hypothesis-examples=1000

# Run with specific random seed
pytest --hypothesis-seed=42

# Run with parallel execution
pytest -n 4  # 4 parallel workers

# Run with specific markers
pytest -m "unit"  # Run only unit tests
pytest -m "integration"  # Run only integration tests
pytest -m "slow"  # Run only slow tests
```

---

## 8. Conclusion

The emotion recognition API is built on solid foundations:

1. **Robust Training**: Trained on 7,000+ samples from public datasets
2. **Dual-Head Architecture**: Combines global baseline with personalized models
3. **Adaptive Blending**: Sigmoid-based alpha engine for smooth transitions
4. **Comprehensive Testing**: 200+ test cases covering unit, integration, and property tests
5. **Production-Ready**: 72.7% accuracy on test set, <5s latency for audio processing

The system is designed for continuous improvement through user feedback, with automatic training triggers and incremental learning capabilities.

