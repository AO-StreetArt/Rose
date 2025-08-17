import hashlib
from typing import Dict, List, Optional, Union, Any
from PIL import Image
import numpy as np
import io
from datetime import datetime, timedelta
import logging
from collections import defaultdict


class MemoryImageStorage:
    """
    In-memory storage system for image parts and metadata.

    This class provides functionality to:
    - Store image parts (segments, crops, features) with metadata in memory
    - Retrieve images by tags, keys, or similarity
    - Manage image lifecycle and cleanup
    - Support both binary and encoded image storage
    - Compatible with RedisImageStorage for seamless data transfer
    """

    def __init__(
        self,
        max_memory_mb: int = 1024,
        ttl_hours: int = 24,
        cleanup_interval: int = 100
    ):
        """
        Initialize in-memory image storage.

        Args:
            max_memory_mb: Maximum memory usage in MB before cleanup
            ttl_hours: Default TTL for stored images in hours
            cleanup_interval: Number of operations before automatic cleanup
        """
        self.max_memory_mb = max_memory_mb
        self.ttl_hours = ttl_hours
        self.cleanup_interval = cleanup_interval
        self.logger = logging.getLogger(__name__)
        self.operation_count = 0

        # In-memory storage structures
        self.images: Dict[str, bytes] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.tags: Dict[str, set] = defaultdict(set)
        self.key_to_tags: Dict[str, List[str]] = {}
        self.expiry_times: Dict[str, datetime] = {}
        self.all_keys: set = set()

        # Key prefixes for compatibility with RedisImageStorage
        self.IMAGE_PREFIX = "img:"
        self.METADATA_PREFIX = "meta:"
        self.TAG_PREFIX = "tag:"
        self.INDEX_PREFIX = "idx:"

    def _generate_image_key(
        self, image_data: Union[np.ndarray, Image.Image, bytes], prefix: str = ""
    ) -> str:
        """Generate a unique key for an image based on its content hash."""
        if isinstance(image_data, np.ndarray):
            # Convert numpy array to bytes for hashing
            image_bytes = image_data.tobytes()
        elif isinstance(image_data, Image.Image):
            # Convert PIL image to bytes
            buffer = io.BytesIO()
            image_data.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()
        else:
            image_bytes = image_data

        # Generate hash from image content
        content_hash = hashlib.sha256(image_bytes).hexdigest()[:16]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # Include microseconds

        if prefix:
            return f"{prefix}_{content_hash}_{timestamp}"
        return f"{content_hash}_{timestamp}"

    def _encode_image(self, image_data: Union[np.ndarray, Image.Image, bytes]) -> bytes:
        """Encode image data to bytes for storage."""
        if isinstance(image_data, np.ndarray):
            # Convert numpy array to PIL Image then to bytes
            if image_data.dtype != np.uint8:
                image_data = (image_data * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_data)
            buffer = io.BytesIO()
            pil_image.save(buffer, format="PNG")
            return buffer.getvalue()
        elif isinstance(image_data, Image.Image):
            buffer = io.BytesIO()
            image_data.save(buffer, format="PNG")
            return buffer.getvalue()
        else:
            return image_data

    def _decode_image(
        self, image_bytes: bytes, target_format: str = "PIL"
    ) -> Union[np.ndarray, Image.Image]:
        """Decode stored image bytes to the target format."""
        if target_format == "PIL":
            return Image.open(io.BytesIO(image_bytes))
        elif target_format == "numpy":
            pil_image = Image.open(io.BytesIO(image_bytes))
            return np.array(pil_image)
        else:
            raise ValueError(f"Unsupported target format: {target_format}")

    def _check_memory_limit(self) -> None:
        """Check if memory usage exceeds limit and cleanup if necessary."""
        current_memory_mb = sum(len(img) for img in self.images.values()) / (1024 * 1024)

        if current_memory_mb > self.max_memory_mb:
            self.logger.warning(f"Memory limit exceeded ({current_memory_mb:.2f}MB), cleaning up...")
            self._cleanup_expired()

    def _cleanup_expired(self) -> None:
        """Remove expired images and metadata."""
        current_time = datetime.now()
        expired_keys = []

        for key, expiry_time in self.expiry_times.items():
            if current_time > expiry_time:
                expired_keys.append(key)

        for key in expired_keys:
            self.delete_image(key)

        if expired_keys:
            self.logger.info(f"Cleaned up {len(expired_keys)} expired images")

    def store_image(
        self,
        image_data: Union[np.ndarray, Image.Image, bytes],
        metadata: Dict[str, Any],
        tags: Optional[List[str]] = None,
        key: Optional[str] = None,
        ttl_hours: Optional[int] = None,
    ) -> str:
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
            # Validate image data
            if not isinstance(image_data, (np.ndarray, Image.Image, bytes)):
                raise ValueError(f"Unsupported image data type: {type(image_data)}")

            # Generate key if not provided
            if key is None:
                key = self._generate_image_key(image_data)

            # Encode image data
            image_bytes = self._encode_image(image_data)

            # Store image data
            self.images[key] = image_bytes

            # Store metadata
            metadata["stored_at"] = datetime.now().isoformat()
            metadata["image_size_bytes"] = len(image_bytes)
            metadata["tags"] = tags or []

            self.metadata[key] = metadata.copy()

            # Set TTL
            ttl = ttl_hours or self.ttl_hours
            if ttl > 0:
                expiry_time = datetime.now().replace(microsecond=0) + timedelta(hours=ttl)
                self.expiry_times[key] = expiry_time

            # Index by tags
            if tags:
                self.key_to_tags[key] = tags.copy()
                for tag in tags:
                    self.tags[tag].add(key)

            # Add to global index
            self.all_keys.add(key)

            # Increment operation count and check memory
            self.operation_count += 1
            if self.operation_count % self.cleanup_interval == 0:
                self._check_memory_limit()

            # Also check memory limit if we're close to the limit
            current_memory_mb = len(image_bytes) / (1024 * 1024)
            if current_memory_mb > self.max_memory_mb * 0.8:  # Check at 80% of limit
                self._check_memory_limit()

            self.logger.info(f"Stored image with key: {key}")
            return key

        except Exception as e:
            self.logger.error(f"Failed to store image: {e}")
            raise

    def retrieve_image(
        self, key: str, format: str = "PIL"
    ) -> Optional[Union[np.ndarray, Image.Image]]:
        """
        Retrieve an image by its key.

        Args:
            key: The key of the stored image
            format: Desired output format ('PIL' or 'numpy')

        Returns:
            Image data in the requested format, or None if not found
        """
        try:
            if key not in self.images:
                self.logger.warning(f"Image not found with key: {key}")
                return None

            # Check if expired
            if key in self.expiry_times and datetime.now() > self.expiry_times[key]:
                self.delete_image(key)
                return None

            image_bytes = self.images[key]
            return self._decode_image(image_bytes, format)

        except ValueError as e:
            # Re-raise ValueError for invalid formats
            raise e
        except Exception as e:
            self.logger.error(f"Failed to retrieve image {key}: {e}")
            return None

    def get_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve metadata for a stored image."""
        try:
            if key not in self.metadata:
                return None

            # Check if expired
            if key in self.expiry_times and datetime.now() > self.expiry_times[key]:
                self.delete_image(key)
                return None

            return self.metadata[key].copy()

        except Exception as e:
            self.logger.error(f"Failed to retrieve metadata for {key}: {e}")
            return None

    def update_metadata(self, key: str, updates: Dict[str, Any]) -> bool:
        """
        Update metadata for a stored image.
        
        Args:
            key: The key of the stored image
            updates: Dictionary containing metadata updates
            
        Returns:
            bool: True if update was successful, False otherwise
        """
        try:
            if key not in self.metadata:
                return False

            # Check if expired
            if key in self.expiry_times and datetime.now() > self.expiry_times[key]:
                self.delete_image(key)
                return False

            # Update metadata with new values
            self.metadata[key].update(updates)
            
            # Update timestamp
            self.metadata[key]["updated_at"] = datetime.now().isoformat()
            
            self.logger.info(f"Updated metadata for key: {key}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to update metadata for {key}: {e}")
            return False

    def search_by_tags(
        self, tags: List[str], operator: str = "OR"
    ) -> List[Dict[str, Any]]:
        """
        Search for images by tags.

        Args:
            tags: List of tags to search for
            operator: 'OR' (any tag matches) or 'AND' (all tags must match)

        Returns:
            List of dictionaries containing image keys and metadata
        """
        try:
            if operator == "OR":
                # Union of all tag sets
                result_keys = set()
                for tag in tags:
                    if tag in self.tags:
                        result_keys.update(self.tags[tag])
            elif operator == "AND":
                # Intersection of all tag sets
                result_keys = set()
                if tags:
                    # Start with the first tag's keys
                    first_tag = tags[0]
                    if first_tag in self.tags:
                        result_keys = self.tags[first_tag].copy()
                    else:
                        return []  # First tag doesn't exist, no results

                    # Intersect with remaining tags
                    for tag in tags[1:]:
                        if tag in self.tags:
                            result_keys = result_keys.intersection(self.tags[tag])
                        else:
                            return []  # Tag doesn't exist, no results
            else:
                raise ValueError("Operator must be 'OR' or 'AND'")

            # Retrieve metadata for found keys
            results = []
            for key in result_keys:
                metadata = self.get_metadata(key)
                if metadata:
                    results.append({"key": key, "metadata": metadata})

            return results

        except ValueError as e:
            # Re-raise ValueError for invalid operators
            raise
        except Exception as e:
            self.logger.error(f"Failed to search by tags: {e}")
            return []

    def list_all_images(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List all stored images with their metadata."""
        try:
            results = []
            count = 0

            for key in list(self.all_keys):
                if count >= limit:
                    break

                metadata = self.get_metadata(key)
                if metadata:
                    results.append({"key": key, "metadata": metadata})
                    count += 1

            return results

        except Exception as e:
            self.logger.error(f"Failed to list images: {e}")
            return []

    def delete_image(self, key: str) -> bool:
        """Delete an image and its associated data."""
        try:
            # Check if key exists
            if key not in self.images:
                return False

            # Remove from tag sets
            if key in self.key_to_tags:
                for tag in self.key_to_tags[key]:
                    if tag in self.tags:
                        self.tags[tag].discard(key)
                        if not self.tags[tag]:  # Remove empty tag sets
                            del self.tags[tag]

            # Remove from global index
            self.all_keys.discard(key)

            # Delete image and metadata
            self.images.pop(key, None)
            self.metadata.pop(key, None)
            self.key_to_tags.pop(key, None)
            self.expiry_times.pop(key, None)

            self.logger.info(f"Deleted image with key: {key}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to delete image {key}: {e}")
            return False

    def cleanup_expired(self) -> int:
        """Clean up expired images and return count of cleaned items."""
        try:
            cleaned_count = 0
            current_time = datetime.now()

            for key in list(self.expiry_times.keys()):
                if current_time > self.expiry_times[key]:
                    if self.delete_image(key):
                        cleaned_count += 1

            self.logger.info(f"Cleaned up {cleaned_count} expired items")
            return cleaned_count

        except Exception as e:
            self.logger.error(f"Failed to cleanup expired items: {e}")
            return 0

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics and memory usage information."""
        try:
            total_images = len(self.all_keys)
            total_size_bytes = sum(len(img) for img in self.images.values())
            tag_counts = {}

            for tag, keys in self.tags.items():
                tag_counts[tag] = len(keys)

            return {
                "total_images": total_images,
                "total_size_mb": total_size_bytes / (1024 * 1024),
                "tag_counts": tag_counts,
                "memory_usage_mb": total_size_bytes / (1024 * 1024),
                "operation_count": self.operation_count,
            }

        except Exception as e:
            self.logger.error(f"Failed to get storage stats: {e}")
            return {}

    def transfer_to_redis(self, redis_storage) -> int:
        """
        Transfer all images from memory to Redis storage.

        Args:
            redis_storage: RedisImageStorage instance

        Returns:
            int: Number of images transferred
        """
        try:
            transferred_count = 0

            for key in list(self.all_keys):
                if key in self.images and key in self.metadata:
                    try:
                        # Retrieve image data
                        image_data = self.retrieve_image(key, "PIL")
                        if image_data is not None:
                            # Store in Redis
                            redis_storage.store_image(
                                image_data=image_data,
                                metadata=self.metadata[key],
                                tags=self.key_to_tags.get(key, []),
                                key=key,
                                ttl_hours=self.ttl_hours
                            )
                            transferred_count += 1
                    except Exception as e:
                        self.logger.error(f"Failed to transfer image {key}: {e}")
                        continue

            self.logger.info(f"Transferred {transferred_count} images to Redis")
            return transferred_count

        except Exception as e:
            self.logger.error(f"Failed to transfer images to Redis: {e}")
            return 0

    def clear_all(self) -> int:
        """Clear all stored images and return count of cleared items."""
        try:
            cleared_count = len(self.all_keys)

            self.images.clear()
            self.metadata.clear()
            self.tags.clear()
            self.key_to_tags.clear()
            self.expiry_times.clear()
            self.all_keys.clear()
            self.operation_count = 0

            self.logger.info(f"Cleared {cleared_count} images from memory")
            return cleared_count

        except Exception as e:
            self.logger.error(f"Failed to clear all images: {e}")
            return 0

    def __len__(self) -> int:
        """Return the number of stored images."""
        return len(self.all_keys)

    def __contains__(self, key: str) -> bool:
        """Check if an image key exists."""
        return key in self.all_keys
