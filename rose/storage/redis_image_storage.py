import redis
import json
import pickle
import hashlib
from typing import Dict, List, Optional, Union, Tuple, Any
from PIL import Image
import numpy as np
import io
import base64
from datetime import datetime, timedelta
import logging


class RedisImageStorage:
    """
    Redis-based storage system for image parts and metadata.
    
    This class provides functionality to:
    - Store image parts (segments, crops, features) with metadata
    - Retrieve images by tags, keys, or similarity
    - Manage image lifecycle and cleanup
    - Support both binary and encoded image storage
    """
    
    def __init__(self, 
                 host: str = 'localhost',
                 port: int = 6379,
                 db: int = 0,
                 password: Optional[str] = None,
                 decode_responses: bool = False,
                 max_memory_mb: int = 1024,
                 ttl_hours: int = 24):
        """
        Initialize Redis image storage.
        
        Args:
            host: Redis server hostname
            port: Redis server port
            db: Redis database number
            password: Redis password if authentication is required
            decode_responses: Whether to decode Redis responses
            max_memory_mb: Maximum memory usage in MB before cleanup
            ttl_hours: Default TTL for stored images in hours
        """
        self.redis_client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=decode_responses
        )
        
        self.max_memory_mb = max_memory_mb
        self.ttl_hours = ttl_hours
        self.logger = logging.getLogger(__name__)
        
        # Key prefixes for different data types
        self.IMAGE_PREFIX = "img:"
        self.METADATA_PREFIX = "meta:"
        self.TAG_PREFIX = "tag:"
        self.INDEX_PREFIX = "idx:"
        
        # Test connection
        try:
            self.redis_client.ping()
            self.logger.info(f"Connected to Redis at {host}:{port}")
        except redis.ConnectionError as e:
            self.logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def _generate_image_key(self, image_data: Union[np.ndarray, Image.Image, bytes], 
                           prefix: str = "") -> str:
        """Generate a unique key for an image based on its content hash."""
        if isinstance(image_data, np.ndarray):
            # Convert numpy array to bytes for hashing
            image_bytes = image_data.tobytes()
        elif isinstance(image_data, Image.Image):
            # Convert PIL image to bytes
            buffer = io.BytesIO()
            image_data.save(buffer, format='PNG')
            image_bytes = buffer.getvalue()
        else:
            image_bytes = image_data
        
        # Generate hash from image content
        content_hash = hashlib.sha256(image_bytes).hexdigest()[:16]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if prefix:
            return f"{self.IMAGE_PREFIX}{prefix}_{content_hash}_{timestamp}"
        return f"{self.IMAGE_PREFIX}{content_hash}_{timestamp}"
    
    def _encode_image(self, image_data: Union[np.ndarray, Image.Image, bytes]) -> bytes:
        """Encode image data to bytes for storage."""
        if isinstance(image_data, np.ndarray):
            # Convert numpy array to PIL Image then to bytes
            if image_data.dtype != np.uint8:
                image_data = (image_data * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_data)
            buffer = io.BytesIO()
            pil_image.save(buffer, format='PNG')
            return buffer.getvalue()
        elif isinstance(image_data, Image.Image):
            buffer = io.BytesIO()
            image_data.save(buffer, format='PNG')
            return buffer.getvalue()
        else:
            return image_data
    
    def _decode_image(self, image_bytes: bytes, target_format: str = 'PIL') -> Union[np.ndarray, Image.Image]:
        """Decode stored image bytes to the target format."""
        if target_format == 'PIL':
            return Image.open(io.BytesIO(image_bytes))
        elif target_format == 'numpy':
            pil_image = Image.open(io.BytesIO(image_bytes))
            return np.array(pil_image)
        else:
            raise ValueError(f"Unsupported target format: {target_format}")
    
    def store_image(self, 
                   image_data: Union[np.ndarray, Image.Image, bytes],
                   metadata: Dict[str, Any],
                   tags: Optional[List[str]] = None,
                   key: Optional[str] = None,
                   ttl_hours: Optional[int] = None) -> str:
        """
        Store an image with metadata and tags.
        
        Args:
            image_data: Image data (numpy array, PIL Image, or bytes)
            metadata: Dictionary containing image metadata
            tags: List of tags for categorization
            key: Custom key for the image (auto-generated if None)
            ttl_hours: Time to live in hours (uses default if None)
            
        Returns:
            str: The key used to store the image
        """
        try:
            # Generate key if not provided
            if key is None:
                key = self._generate_image_key(image_data)
            
            # Encode image data
            image_bytes = self._encode_image(image_data)
            
            # Store image data
            self.redis_client.set(f"{self.IMAGE_PREFIX}{key}", image_bytes)
            
            # Store metadata
            metadata['stored_at'] = datetime.now().isoformat()
            metadata['image_size_bytes'] = len(image_bytes)
            metadata['tags'] = tags or []
            
            self.redis_client.set(f"{self.METADATA_PREFIX}{key}", 
                                json.dumps(metadata, default=str))
            
            # Set TTL
            ttl = ttl_hours or self.ttl_hours
            if ttl > 0:
                self.redis_client.expire(f"{self.IMAGE_PREFIX}{key}", ttl * 3600)
                self.redis_client.expire(f"{self.METADATA_PREFIX}{key}", ttl * 3600)
            
            # Index by tags
            if tags:
                for tag in tags:
                    tag_key = f"{self.TAG_PREFIX}{tag}"
                    self.redis_client.sadd(tag_key, key)
                    if ttl > 0:
                        self.redis_client.expire(tag_key, ttl * 3600)
            
            # Add to global index
            self.redis_client.sadd(f"{self.INDEX_PREFIX}all", key)
            
            self.logger.info(f"Stored image with key: {key}")
            return key
            
        except Exception as e:
            self.logger.error(f"Failed to store image: {e}")
            raise
    
    def retrieve_image(self, key: str, format: str = 'PIL') -> Optional[Union[np.ndarray, Image.Image]]:
        """
        Retrieve an image by its key.
        
        Args:
            key: The key of the stored image
            format: Desired output format ('PIL' or 'numpy')
            
        Returns:
            Image data in the requested format, or None if not found
        """
        try:
            image_key = f"{self.IMAGE_PREFIX}{key}"
            image_bytes = self.redis_client.get(image_key)
            
            if image_bytes is None:
                self.logger.warning(f"Image not found with key: {key}")
                return None
            
            return self._decode_image(image_bytes, format)
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve image {key}: {e}")
            return None
    
    def get_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve metadata for a stored image."""
        try:
            metadata_key = f"{self.METADATA_PREFIX}{key}"
            metadata_json = self.redis_client.get(metadata_key)
            
            if metadata_json is None:
                return None
            
            return json.loads(metadata_json)
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve metadata for {key}: {e}")
            return None
    
    def search_by_tags(self, tags: List[str], 
                      operator: str = 'OR') -> List[Dict[str, Any]]:
        """
        Search for images by tags.
        
        Args:
            tags: List of tags to search for
            operator: 'OR' (any tag matches) or 'AND' (all tags must match)
            
        Returns:
            List of dictionaries containing image keys and metadata
        """
        try:
            if operator == 'OR':
                # Union of all tag sets
                result_keys = set()
                for tag in tags:
                    tag_key = f"{self.TAG_PREFIX}{tag}"
                    keys = self.redis_client.smembers(tag_key)
                    result_keys.update(keys)
            elif operator == 'AND':
                # Intersection of all tag sets
                result_keys = None
                for tag in tags:
                    tag_key = f"{self.TAG_PREFIX}{tag}"
                    keys = self.redis_client.smembers(tag_key)
                    if result_keys is None:
                        result_keys = keys
                    else:
                        result_keys = result_keys.intersection(keys)
            else:
                raise ValueError("Operator must be 'OR' or 'AND'")
            
            # Retrieve metadata for found keys
            results = []
            for key in result_keys:
                metadata = self.get_metadata(key)
                if metadata:
                    results.append({
                        'key': key,
                        'metadata': metadata
                    })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to search by tags: {e}")
            return []
    
    def list_all_images(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List all stored images with their metadata."""
        try:
            all_keys = self.redis_client.smembers(f"{self.INDEX_PREFIX}all")
            results = []
            
            for key in list(all_keys)[:limit]:
                metadata = self.get_metadata(key)
                if metadata:
                    results.append({
                        'key': key,
                        'metadata': metadata
                    })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to list images: {e}")
            return []
    
    def delete_image(self, key: str) -> bool:
        """Delete an image and its associated data."""
        try:
            # Get metadata to find tags
            metadata = self.get_metadata(key)
            
            # Remove from tag sets
            if metadata and 'tags' in metadata:
                for tag in metadata['tags']:
                    tag_key = f"{self.TAG_PREFIX}{tag}"
                    self.redis_client.srem(tag_key, key)
            
            # Remove from global index
            self.redis_client.srem(f"{self.INDEX_PREFIX}all", key)
            
            # Delete image and metadata
            self.redis_client.delete(f"{self.IMAGE_PREFIX}{key}")
            self.redis_client.delete(f"{self.METADATA_PREFIX}{key}")
            
            self.logger.info(f"Deleted image with key: {key}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete image {key}: {e}")
            return False
    
    def cleanup_expired(self) -> int:
        """Clean up expired images and return count of cleaned items."""
        try:
            # Redis automatically handles TTL cleanup, but we can clean up orphaned metadata
            cleaned_count = 0
            
            # Get all keys and check for orphaned metadata
            all_keys = self.redis_client.smembers(f"{self.INDEX_PREFIX}all")
            
            for key in all_keys:
                image_exists = self.redis_client.exists(f"{self.IMAGE_PREFIX}{key}")
                if not image_exists:
                    # Clean up orphaned metadata and tag references
                    metadata = self.get_metadata(key)
                    if metadata and 'tags' in metadata:
                        for tag in metadata['tags']:
                            tag_key = f"{self.TAG_PREFIX}{tag}"
                            self.redis_client.srem(tag_key, key)
                    
                    self.redis_client.delete(f"{self.METADATA_PREFIX}{key}")
                    self.redis_client.srem(f"{self.INDEX_PREFIX}all", key)
                    cleaned_count += 1
            
            self.logger.info(f"Cleaned up {cleaned_count} expired/orphaned items")
            return cleaned_count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup expired items: {e}")
            return 0
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics and memory usage information."""
        try:
            all_keys = self.redis_client.smembers(f"{self.INDEX_PREFIX}all")
            total_images = len(all_keys)
            
            total_size_bytes = 0
            tag_counts = {}
            
            for key in all_keys:
                metadata = self.get_metadata(key)
                if metadata and 'image_size_bytes' in metadata:
                    total_size_bytes += metadata['image_size_bytes']
                
                if metadata and 'tags' in metadata:
                    for tag in metadata['tags']:
                        tag_counts[tag] = tag_counts.get(tag, 0) + 1
            
            # Get Redis info
            info = self.redis_client.info()
            
            return {
                'total_images': total_images,
                'total_size_mb': total_size_bytes / (1024 * 1024),
                'tag_counts': tag_counts,
                'redis_memory_usage_mb': info.get('used_memory', 0) / (1024 * 1024),
                'redis_keyspace_hits': info.get('keyspace_hits', 0),
                'redis_keyspace_misses': info.get('keyspace_misses', 0)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get storage stats: {e}")
            return {}
    
    def close(self):
        """Close the Redis connection."""
        try:
            self.redis_client.close()
            self.logger.info("Redis connection closed")
        except Exception as e:
            self.logger.error(f"Error closing Redis connection: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
