"""
PyTorch MLP model for speech emotion classification.

Architecture
------------
Input(419) → Linear(256) → BN → ReLU → Dropout(0.3)
          → Linear(128) → BN → ReLU → Dropout(0.3)
          → Linear(64)  → ReLU
          → Linear(8)                   (logits — softmax applied externally)

Total trainable parameters: ~150 K.
Designed for fast CPU inference in a FastAPI request path.
"""

import torch
import torch.nn as nn

# Import shared config so the model always matches the feature pipeline.
import sys, pathlib

# Allow imports from api/app when running training scripts directly
_api_root = pathlib.Path(__file__).resolve().parents[1]
if str(_api_root) not in sys.path:
    sys.path.insert(0, str(_api_root))

from app.services.feature_config import FEATURE_DIM, NUM_CLASSES


class EmotionMLP(nn.Module):
    """
    Feed-forward neural network for emotion classification.

    Parameters
    ----------
    input_dim : int
        Length of the feature vector (default from ``feature_config.FEATURE_DIM``).
    num_classes : int
        Number of emotion categories (default from ``feature_config.NUM_CLASSES``).
    dropout : float
        Dropout probability applied after each hidden layer.
    """

    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (batch, input_dim)

        Returns
        -------
        Tensor of shape (batch, num_classes)  — raw logits (no softmax).
        """
        return self.net(x)
