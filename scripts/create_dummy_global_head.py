"""
Create a dummy global emotion head for testing purposes.
WARNING: This creates a model with random weights - NOT for production use!
"""
import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add api to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.global_emotion_head import GlobalEmotionHead
from app.services.feature_config import EMBEDDING_DIM, NUM_CLASSES, MODEL_DIR

def create_dummy_model():
    """Create and save a dummy global emotion head with random weights."""
    
    # Create model with random initialization
    model = GlobalEmotionHead(embedding_dim=EMBEDDING_DIM, num_classes=NUM_CLASSES)
    
    # Ensure models directory exists
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save model
    save_path = MODEL_DIR / "global_emotion_head.pt"
    torch.save(model.state_dict(), str(save_path))
    
    print(f"✓ Dummy global emotion head created at: {save_path}")
    print(f"  Model architecture: Linear({EMBEDDING_DIM} → {NUM_CLASSES})")
    print(f"  Weight shape: {model.linear.weight.shape}")
    print(f"  Bias shape: {model.linear.bias.shape}")
    print()
    print("⚠️  WARNING: This model has RANDOM weights and will give random predictions!")
    print("⚠️  For real emotion recognition, train the model properly using:")
    print("    python -m training.train_wav2vec2 --save-as-global-head")

if __name__ == "__main__":
    create_dummy_model()
