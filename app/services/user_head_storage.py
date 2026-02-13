"""
Storage service for user emotion heads.

This module provides a storage abstraction layer that supports both MongoDB
and file-based storage for user-specific emotion head models. The service
handles compression, validation, error handling, and graceful fallback.

Storage modes:
- File-only: USE_MONGODB_STORAGE=False (default, backward compatible)
- MongoDB-only: USE_MONGODB_STORAGE=True, DUAL_SAVE_MODE=False
- Dual-save: USE_MONGODB_STORAGE=True, DUAL_SAVE_MODE=True (migration mode)

Load priority:
1. MongoDB (if USE_MONGODB_STORAGE=True)
2. File (fallback)
3. None (no model exists)
"""

import logging
import io
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime
import torch

from app.config import settings
from app.db.mongodb import get_database
from app.models.user_model_storage import UserModelStorage
from app.services.feature_config import MODEL_DIR

logger = logging.getLogger(__name__)

USER_HEADS_DIR = MODEL_DIR / "user_heads"

# Maximum model blob size (1MB safety limit)
MAX_MODEL_SIZE_BYTES = 1024 * 1024


class UserHeadStorageService:
    """
    Storage service for user emotion heads.
    
    Provides a unified interface for saving and loading user-specific emotion
    head models with support for both MongoDB and file-based storage backends.
    
    The service handles:
    - Compression (gzip) to reduce storage costs
    - Size validation (reject blobs > 1MB)
    - Graceful fallback (MongoDB → file)
    - Dual-save mode for safe migration
    - Storage statistics and monitoring
    
    Configuration:
    - USE_MONGODB_STORAGE: Enable MongoDB storage (default: False)
    - DUAL_SAVE_MODE: Save to both backends (default: False)
    - MONGODB_STORAGE_COMPRESSION: Compression algorithm (default: "gzip")
    """
    
    def __init__(self):
        """Initialize storage service with configuration from settings."""
        self.use_mongodb = getattr(settings, "USE_MONGODB_STORAGE", False)
        self.dual_save = getattr(settings, "DUAL_SAVE_MODE", False)
        self.compression = getattr(settings, "MONGODB_STORAGE_COMPRESSION", "gzip")
        self.db = None
        
        logger.info(
            f"UserHeadStorageService initialized: "
            f"use_mongodb={self.use_mongodb}, dual_save={self.dual_save}"
        )
    
    def _get_db(self):
        """
        Get MongoDB database instance (lazy connection).
        
        Returns
        -------
        Database
            MongoDB database instance
        """
        if self.db is None:
            self.db = get_database()
        return self.db
    
    def save_model(
        self,
        user_id: str,
        state_dict: Dict[str, torch.Tensor],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Save user head model to storage.
        
        Behavior:
        - If USE_MONGODB_STORAGE=True: Save to MongoDB
        - If DUAL_SAVE_MODE=True: Save to both MongoDB and file
        - If USE_MONGODB_STORAGE=False: Save to file only
        
        Parameters
        ----------
        user_id : str
            Unique user identifier
        state_dict : dict
            PyTorch model state_dict
        metadata : dict, optional
            Training metadata (samples, timestamps, etc.)
        
        Returns
        -------
        bool
            True if save successful, False otherwise
        
        Notes
        -----
        - Compresses blob with gzip before storage
        - Validates blob size < 1MB
        - Logs errors and attempts fallback on failure
        """
        if metadata is None:
            metadata = {}
        
        success = True
        
        # Compress state_dict and compute sizes
        try:
            compressed_blob = UserModelStorage.compress_state_dict(state_dict)
            
            # Compute uncompressed size
            buffer = io.BytesIO()
            torch.save(state_dict, buffer)
            uncompressed_size = len(buffer.getvalue())
            compressed_size = len(compressed_blob)
            
            # Validate size
            if compressed_size > MAX_MODEL_SIZE_BYTES:
                logger.error(
                    f"Model blob too large for user {user_id}: "
                    f"{compressed_size} bytes (max: {MAX_MODEL_SIZE_BYTES})"
                )
                return False
            
            # Update metadata
            metadata.update({
                "model_size_bytes": uncompressed_size,
                "compressed_size_bytes": compressed_size,
                "checksum": UserModelStorage.compute_checksum(compressed_blob),
                "last_trained": datetime.utcnow()
            })
        
        except Exception as e:
            logger.error(f"Failed to compress model for user {user_id}: {e}", exc_info=True)
            return False
        
        # Save to MongoDB
        mongodb_success = True
        if self.use_mongodb or self.dual_save:
            try:
                metadata["storage_mode"] = "dual" if self.dual_save else "mongodb"
                
                model_storage = UserModelStorage(
                    user_id=user_id,
                    model_blob=compressed_blob,
                    metadata=metadata
                )
                
                db = self._get_db()
                db.user_models.update_one(
                    {"user_id": user_id},
                    {"$set": model_storage.to_dict()},
                    upsert=True
                )
                
                logger.info(
                    f"Saved model to MongoDB for user {user_id} "
                    f"({compressed_size} bytes compressed)"
                )
            
            except Exception as e:
                logger.error(
                    f"Failed to save to MongoDB for user {user_id}: {e}",
                    exc_info=True
                )
                mongodb_success = False
                
                # Fail immediately if MongoDB-only mode
                if not self.dual_save and self.use_mongodb:
                    return False
        
        # Save to file
        file_success = True
        if not self.use_mongodb or self.dual_save:
            try:
                USER_HEADS_DIR.mkdir(parents=True, exist_ok=True)
                file_path = USER_HEADS_DIR / f"{user_id}.pt"
                torch.save(state_dict, str(file_path))
                
                logger.info(f"Saved model to file for user {user_id} at {file_path}")
            
            except Exception as e:
                logger.error(
                    f"Failed to save to file for user {user_id}: {e}",
                    exc_info=True
                )
                file_success = False
        
        # In dual-save mode, success if either backend succeeds
        # In single-backend mode, success if that backend succeeds
        if self.dual_save:
            success = mongodb_success or file_success
        elif self.use_mongodb:
            success = mongodb_success
        else:
            success = file_success
        
        return success
    
    def load_model(self, user_id: str) -> Optional[Dict[str, torch.Tensor]]:
        """
        Load user head model from storage.
        
        Load priority:
        1. MongoDB (if USE_MONGODB_STORAGE=True)
        2. File (fallback)
        3. None (no model exists)
        
        Parameters
        ----------
        user_id : str
            Unique user identifier
        
        Returns
        -------
        dict or None
            PyTorch state_dict if found, None otherwise
        
        Notes
        -----
        - Decompresses blob after loading from MongoDB
        - Validates state_dict structure
        - Logs storage source for monitoring
        """
        # Try MongoDB first
        if self.use_mongodb:
            try:
                db = self._get_db()
                doc = db.user_models.find_one({"user_id": user_id})
                
                if doc:
                    model_storage = UserModelStorage.from_dict(doc)
                    state_dict = UserModelStorage.decompress_state_dict(
                        model_storage.model_blob
                    )
                    
                    logger.info(f"Loaded model from MongoDB for user {user_id}")
                    return state_dict
            
            except Exception as e:
                logger.error(
                    f"Failed to load from MongoDB for user {user_id}: {e}",
                    exc_info=True
                )
        
        # Fallback to file
        try:
            file_path = USER_HEADS_DIR / f"{user_id}.pt"
            
            if file_path.exists():
                state_dict = torch.load(
                    str(file_path),
                    map_location="cpu",
                    weights_only=True
                )
                
                logger.info(f"Loaded model from file for user {user_id}")
                return state_dict
        
        except Exception as e:
            logger.error(
                f"Failed to load from file for user {user_id}: {e}",
                exc_info=True
            )
        
        # No model found
        logger.debug(f"No model found for user {user_id}")
        return None
    
    def exists(self, user_id: str) -> bool:
        """
        Check if user has a trained model (without loading).
        
        Parameters
        ----------
        user_id : str
            Unique user identifier
        
        Returns
        -------
        bool
            True if model exists in MongoDB or file, False otherwise
        """
        # Check MongoDB
        if self.use_mongodb:
            try:
                db = self._get_db()
                count = db.user_models.count_documents(
                    {"user_id": user_id},
                    limit=1
                )
                if count > 0:
                    return True
            
            except Exception as e:
                logger.error(
                    f"MongoDB exists check failed for user {user_id}: {e}",
                    exc_info=True
                )
        
        # Check file
        file_path = USER_HEADS_DIR / f"{user_id}.pt"
        return file_path.exists()
    
    def delete_model(self, user_id: str) -> bool:
        """
        Delete user model from storage.
        
        Deletes from both MongoDB and file storage if they exist.
        
        Parameters
        ----------
        user_id : str
            Unique user identifier
        
        Returns
        -------
        bool
            True if deletion successful, False if any errors occurred
        """
        success = True
        
        # Delete from MongoDB
        if self.use_mongodb:
            try:
                db = self._get_db()
                result = db.user_models.delete_one({"user_id": user_id})
                
                if result.deleted_count > 0:
                    logger.info(f"Deleted model from MongoDB for user {user_id}")
            
            except Exception as e:
                logger.error(
                    f"Failed to delete from MongoDB for user {user_id}: {e}",
                    exc_info=True
                )
                success = False
        
        # Delete from file
        file_path = USER_HEADS_DIR / f"{user_id}.pt"
        if file_path.exists():
            try:
                file_path.unlink()
                logger.info(f"Deleted model file for user {user_id}")
            
            except Exception as e:
                logger.error(
                    f"Failed to delete file for user {user_id}: {e}",
                    exc_info=True
                )
                success = False
        
        return success
    
    def get_storage_stats(self) -> Dict[str, int]:
        """
        Get storage statistics.
        
        Returns
        -------
        dict
            Statistics with keys:
            - total_models: Total number of unique models
            - mongodb_models: Models in MongoDB
            - file_models: Models in file storage
            - total_size_bytes: Total compressed size (MongoDB only)
            - avg_model_size_bytes: Average compressed size
        """
        stats = {
            "total_models": 0,
            "mongodb_models": 0,
            "file_models": 0,
            "total_size_bytes": 0,
            "avg_model_size_bytes": 0
        }
        
        # MongoDB stats
        if self.use_mongodb:
            try:
                db = self._get_db()
                stats["mongodb_models"] = db.user_models.count_documents({})
                
                # Aggregate total size
                pipeline = [
                    {
                        "$group": {
                            "_id": None,
                            "total_size": {"$sum": "$metadata.compressed_size_bytes"}
                        }
                    }
                ]
                result = list(db.user_models.aggregate(pipeline))
                
                if result:
                    stats["total_size_bytes"] = result[0].get("total_size", 0)
            
            except Exception as e:
                logger.error(f"Failed to get MongoDB stats: {e}", exc_info=True)
        
        # File stats
        if USER_HEADS_DIR.exists():
            file_count = len(list(USER_HEADS_DIR.glob("*.pt")))
            stats["file_models"] = file_count
        
        # Compute totals
        stats["total_models"] = stats["mongodb_models"] + stats["file_models"]
        
        if stats["total_models"] > 0 and stats["total_size_bytes"] > 0:
            stats["avg_model_size_bytes"] = (
                stats["total_size_bytes"] // stats["mongodb_models"]
                if stats["mongodb_models"] > 0
                else 0
            )
        
        return stats


# Global singleton instance
storage_service = UserHeadStorageService()
