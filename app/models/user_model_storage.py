"""
MongoDB model for user emotion head storage.

This module provides the UserModelStorage class for storing and retrieving
user-specific emotion head models in MongoDB. Models are stored as compressed
binary blobs with metadata for integrity verification and auditing.

Storage format:
- model_blob: gzip-compressed PyTorch state_dict (Binary)
- metadata: training info, checksums, sizes
- indexes: user_id (unique), updated_at (for cleanup)
"""

from datetime import datetime
from typing import Optional, Dict, Any
from bson import ObjectId, Binary
import gzip
import hashlib
import torch
import io
import logging

logger = logging.getLogger(__name__)


class UserModelStorage:
    """
    MongoDB document model for user emotion head storage.
    
    This class handles serialization, compression, and integrity verification
    for user-specific emotion head models stored in MongoDB.
    
    Attributes
    ----------
    _id : ObjectId
        MongoDB document ID
    user_id : str
        Unique user identifier
    model_blob : bytes
        gzip-compressed PyTorch state_dict
    version : int
        Model version (for future versioning support)
    created_at : datetime
        Timestamp of first model save
    updated_at : datetime
        Timestamp of last model update
    metadata : dict
        Training metadata (samples, sizes, checksums)
    """
    
    def __init__(
        self,
        user_id: str,
        model_blob: bytes,
        version: int = 1,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
        _id: Optional[ObjectId] = None
    ):
        """
        Initialize UserModelStorage instance.
        
        Parameters
        ----------
        user_id : str
            Unique user identifier
        model_blob : bytes
            gzip-compressed PyTorch state_dict
        version : int, default=1
            Model version number
        created_at : datetime, optional
            Creation timestamp (defaults to now)
        updated_at : datetime, optional
            Update timestamp (defaults to now)
        metadata : dict, optional
            Training metadata
        _id : ObjectId, optional
            MongoDB document ID (auto-generated if None)
        """
        self._id = _id or ObjectId()
        self.user_id = user_id
        self.model_blob = model_blob
        self.version = version
        self.created_at = created_at or datetime.utcnow()
        self.updated_at = updated_at or datetime.utcnow()
        self.metadata = metadata or {}
    
    @staticmethod
    def compress_state_dict(state_dict: Dict[str, torch.Tensor]) -> bytes:
        """
        Compress PyTorch state_dict to gzip bytes.
        
        Parameters
        ----------
        state_dict : dict
            PyTorch model state_dict
        
        Returns
        -------
        bytes
            gzip-compressed binary data
        
        Raises
        ------
        Exception
            If compression fails
        """
        try:
            # Serialize state_dict to bytes
            buffer = io.BytesIO()
            torch.save(state_dict, buffer)
            uncompressed = buffer.getvalue()
            
            # Compress with gzip
            compressed = gzip.compress(uncompressed, compresslevel=6)
            
            logger.debug(
                f"Compressed state_dict: {len(uncompressed)} → {len(compressed)} bytes "
                f"({100 * len(compressed) / len(uncompressed):.1f}%)"
            )
            
            return compressed
        
        except Exception as e:
            logger.error(f"Failed to compress state_dict: {e}")
            raise
    
    @staticmethod
    def decompress_state_dict(compressed_bytes: bytes) -> Dict[str, torch.Tensor]:
        """
        Decompress gzip bytes to PyTorch state_dict.
        
        Parameters
        ----------
        compressed_bytes : bytes
            gzip-compressed binary data
        
        Returns
        -------
        dict
            PyTorch model state_dict
        
        Raises
        ------
        Exception
            If decompression or deserialization fails
        """
        try:
            # Decompress
            uncompressed = gzip.decompress(compressed_bytes)
            
            # Deserialize to state_dict
            buffer = io.BytesIO(uncompressed)
            state_dict = torch.load(buffer, map_location="cpu", weights_only=True)
            
            logger.debug(f"Decompressed state_dict: {len(compressed_bytes)} → {len(uncompressed)} bytes")
            
            return state_dict
        
        except Exception as e:
            logger.error(f"Failed to decompress state_dict: {e}")
            raise
    
    @staticmethod
    def compute_checksum(data: bytes) -> str:
        """
        Compute SHA256 checksum for integrity verification.
        
        Parameters
        ----------
        data : bytes
            Binary data to checksum
        
        Returns
        -------
        str
            Checksum in format "sha256:hexdigest"
        """
        return f"sha256:{hashlib.sha256(data).hexdigest()}"
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to MongoDB document format.
        
        Returns
        -------
        dict
            MongoDB document with Binary-wrapped blob
        """
        return {
            "_id": self._id,
            "user_id": self.user_id,
            "model_blob": Binary(self.model_blob),
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, doc: Dict[str, Any]) -> "UserModelStorage":
        """
        Create UserModelStorage from MongoDB document.
        
        Parameters
        ----------
        doc : dict
            MongoDB document
        
        Returns
        -------
        UserModelStorage
            Reconstructed instance
        """
        return cls(
            user_id=doc["user_id"],
            model_blob=bytes(doc["model_blob"]),
            version=doc.get("version", 1),
            created_at=doc.get("created_at"),
            updated_at=doc.get("updated_at"),
            metadata=doc.get("metadata", {}),
            _id=doc.get("_id")
        )
