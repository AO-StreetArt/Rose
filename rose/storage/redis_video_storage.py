#!/usr/bin/env python3
"""
Redis-based video storage system for storing video files by timestamp and tags.

This module provides functionality to:
- Store video files with metadata and tags in Redis
- Retrieve videos by tags, keys, or timestamp ranges
- Manage video lifecycle and cleanup
- Support various video formats and metadata
"""

import redis
import json
import hashlib
from typing import Dict, List, Optional, Union, Any, BinaryIO
from datetime import datetime, timedelta
import logging
import mimetypes


class RedisVideoStorage:
    """
    Redis-based storage system for video files and metadata.

    This class provides functionality to:
    - Store video files with metadata and tags in Redis
    - Retrieve videos by tags, keys, or timestamp ranges
    - Manage video lifecycle and cleanup
    - Support various video formats
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        decode_responses: bool = False,
        max_memory_mb: int = 2048,
        ttl_hours: int = 48,
        max_video_size_mb: int = 100
    ):
        """
        Initialize Redis video storage.

        Args:
            host: Redis server hostname
            port: Redis server port
            db: Redis database number
            password: Redis password if authentication is required
            decode_responses: Whether to decode Redis responses
            max_memory_mb: Maximum memory usage in MB before cleanup
            ttl_hours: Default TTL for stored videos in hours
            max_video_size_mb: Maximum video file size in MB
        """
        # Use separate Redis clients for binary data (videos) and text data
        # (metadata, tags, etc.)
        self.redis_client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=False,  # Keep binary for video data
        )

        # Text client for metadata, tags, and indices
        self.text_client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=True,  # Decode responses for text data
        )

        self.max_memory_mb = max_memory_mb
        self.ttl_hours = ttl_hours
        self.max_video_size_mb = max_video_size_mb
        self.logger = logging.getLogger(__name__)

        # Key prefixes for different data types
        self.VIDEO_PREFIX = "vid:"
        self.METADATA_PREFIX = "meta:"
        self.TAG_PREFIX = "tag:"
        self.INDEX_PREFIX = "idx:"
        self.TIMESTAMP_PREFIX = "ts:"

        # Test connection
        try:
            self.redis_client.ping()
            self.logger.info(f"Connected to Redis at {host}:{port}")
        except redis.ConnectionError as e:
            self.logger.error(f"Failed to connect to Redis: {e}")
            raise

    def _generate_video_key(
        self, video_data: Union[bytes, BinaryIO], filename: Optional[str] = None, prefix: str = ""
    ) -> str:
        """
        Generate a unique key for a video based on its content hash and filename.
        
        Args:
            video_data: Video data as bytes or file-like object
            filename: Optional filename for the video
            prefix: Optional prefix for the key
            
        Returns:
            str: Unique key for the video
        """
        if hasattr(video_data, 'read'):
            # File-like object
            video_bytes = video_data.read()
            video_data.seek(0)  # Reset file pointer
        else:
            video_bytes = video_data

        # Generate hash from video content
        content_hash = hashlib.sha256(video_bytes).hexdigest()[:16]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Include filename in key if provided
        if filename:
            safe_filename = "".join(c for c in filename if c.isalnum() or c in "._-")[:20]
            key = f"{prefix}{safe_filename}_{content_hash}_{timestamp}"
        else:
            key = f"{prefix}{content_hash}_{timestamp}"
            
        return key

    def _validate_video_data(self, video_data: Union[bytes, BinaryIO]) -> bytes:
        """
        Validate and convert video data to bytes.
        
        Args:
            video_data: Video data as bytes or file-like object
            
        Returns:
            bytes: Video data as bytes
            
        Raises:
            ValueError: If video data is invalid or too large
        """
        if hasattr(video_data, 'read'):
            # File-like object
            video_bytes = video_data.read()
            video_data.seek(0)  # Reset file pointer
        else:
            video_bytes = video_data

        # Check video size
        video_size_mb = len(video_bytes) / (1024 * 1024)
        if video_size_mb > self.max_video_size_mb:
            raise ValueError(
                f"Video size {video_size_mb:.2f}MB exceeds maximum allowed size "
                f"of {self.max_video_size_mb}MB"
            )

        if len(video_bytes) == 0:
            raise ValueError("Video data cannot be empty")

        return video_bytes

    def _detect_mime_type(self, video_data: bytes, filename: Optional[str] = None) -> str:
        """
        Detect MIME type of video data.
        
        Args:
            video_data: Video data as bytes
            filename: Optional filename for MIME type detection
            
        Returns:
            str: MIME type of the video
        """
        if filename:
            mime_type, _ = mimetypes.guess_type(filename)
            if mime_type and mime_type.startswith('video/'):
                return mime_type

        # Try to detect from content (basic check for common video formats)
        if video_data.startswith(b'\x00\x00\x00\x18ftypmp4'):
            return 'video/mp4'
        elif video_data.startswith(b'\x1a\x45\xdf\xa3'):
            return 'video/webm'
        elif video_data.startswith(b'RIFF') and b'AVI ' in video_data[:20]:
            return 'video/x-msvideo'
        elif video_data.startswith(b'\x00\x00\x01\xb3'):
            return 'video/mpeg'
        else:
            return 'video/octet-stream'

    def store_video(
        self,
        video_data: Union[bytes, BinaryIO],
        metadata: Dict[str, Any],
        tags: Optional[List[str]] = None,
        key: Optional[str] = None,
        filename: Optional[str] = None,
        ttl_hours: Optional[int] = None,
    ) -> str:
        """
        Store a video with metadata and tags in Redis.

        Args:
            video_data: Video data as bytes or file-like object
            metadata: Dictionary containing video metadata
            tags: List of tags for categorization
            key: Custom key for the video (auto-generated if None)
            filename: Optional filename for the video
            ttl_hours: Time to live in hours (uses default if None)

        Returns:
            str: The key used to store the video

        Raises:
            ValueError: If video data is invalid or too large
            redis.RedisError: If Redis operation fails
        """
        try:
            # Validate and convert video data
            video_bytes = self._validate_video_data(video_data)

            # Generate key if not provided
            if key is None:
                key = self._generate_video_key(video_bytes, filename, self.VIDEO_PREFIX)

            # Detect MIME type
            mime_type = self._detect_mime_type(video_bytes, filename)

            # Prepare metadata
            metadata["stored_at"] = datetime.now().isoformat()
            metadata["video_size_bytes"] = len(video_bytes)
            metadata["mime_type"] = mime_type
            metadata["filename"] = filename
            metadata["tags"] = tags or []
            metadata["duration"] = metadata.get("duration", 0)  # Duration in seconds
            metadata["resolution"] = metadata.get("resolution", "unknown")
            metadata["fps"] = metadata.get("fps", 0)

            # Store video data
            video_key = f"{self.VIDEO_PREFIX}{key}"
            self.redis_client.set(video_key, video_bytes)

            # Store metadata
            metadata_key = f"{self.METADATA_PREFIX}{key}"
            self.text_client.set(metadata_key, json.dumps(metadata))

            # Set TTL
            ttl = ttl_hours or self.ttl_hours
            if ttl > 0:
                ttl_seconds = ttl * 3600
                self.redis_client.expire(video_key, ttl_seconds)
                self.text_client.expire(metadata_key, ttl_seconds)

            # Index by tags
            if tags:
                for tag in tags:
                    tag_key = f"{self.TAG_PREFIX}{tag}"
                    self.text_client.sadd(tag_key, key)
                    # Set TTL for tag index
                    if ttl > 0:
                        self.text_client.expire(tag_key, ttl_seconds)

            # Index by timestamp
            timestamp_key = f"{self.TIMESTAMP_PREFIX}{key}"
            stored_time = datetime.fromisoformat(metadata["stored_at"])
            timestamp_score = stored_time.timestamp()
            self.text_client.zadd("video_timestamps", {timestamp_key: timestamp_score})

            # Add to global index
            self.text_client.sadd("all_video_keys", key)

            self.logger.info(f"Stored video with key: {key}, size: {len(video_bytes)} bytes")
            return key

        except redis.RedisError as e:
            self.logger.error(f"Redis error storing video: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error storing video: {e}")
            raise

    def retrieve_video(self, key: str, format: str = "bytes") -> Optional[Union[bytes, Dict[str, Any]]]:
        """
        Retrieve a video by its key from Redis.

        Args:
            key: The key of the video to retrieve
            format: Return format ("bytes" for raw video data, "metadata" for metadata only)

        Returns:
            Optional[Union[bytes, Dict[str, Any]]]: Video data or metadata, or None if not found
        """
        try:
            if format == "metadata":
                metadata_key = f"{self.METADATA_PREFIX}{key}"
                metadata_json = self.text_client.get(metadata_key)
                if metadata_json is None:
                    self.logger.warning(f"Video metadata with key {key} not found")
                    return None
                return json.loads(metadata_json)
            else:
                video_key = f"{self.VIDEO_PREFIX}{key}"
                video_data = self.redis_client.get(video_key)
                if video_data is None:
                    self.logger.warning(f"Video with key {key} not found")
                    return None
                return video_data

        except redis.RedisError as e:
            self.logger.error(f"Redis error retrieving video {key}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error retrieving video {key}: {e}")
            return None

    def search_by_tags(self, tags: List[str], operator: str = "OR") -> List[Dict[str, Any]]:
        """
        Search for videos by tags in Redis.

        Args:
            tags: List of tags to search for
            operator: Search operator ("AND" or "OR")

        Returns:
            List[Dict[str, Any]]: List of video metadata matching the search criteria
        """
        try:
            matching_keys = set()

            if operator == "AND":
                # All tags must be present
                if tags:
                    first_tag_key = f"{self.TAG_PREFIX}{tags[0]}"
                    matching_keys = set(self.text_client.smembers(first_tag_key))
                    for tag in tags[1:]:
                        tag_key = f"{self.TAG_PREFIX}{tag}"
                        tag_keys = set(self.text_client.smembers(tag_key))
                        matching_keys &= tag_keys
            else:
                # OR: Any tag can match
                for tag in tags:
                    tag_key = f"{self.TAG_PREFIX}{tag}"
                    tag_keys = set(self.text_client.smembers(tag_key))
                    matching_keys.update(tag_keys)

            results = []
            for key in matching_keys:
                metadata = self.retrieve_video(key, format="metadata")
                if metadata:
                    metadata["key"] = key
                    results.append(metadata)

            return results

        except redis.RedisError as e:
            self.logger.error(f"Redis error searching by tags: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Error searching by tags: {e}")
            return []

    def search_by_timestamp_range(
        self, 
        start_time: Optional[datetime] = None, 
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for videos within a timestamp range in Redis.

        Args:
            start_time: Start time for the range (inclusive)
            end_time: End time for the range (inclusive)

        Returns:
            List[Dict[str, Any]]: List of video metadata within the timestamp range
        """
        try:
            results = []
            
            # Convert timestamps to scores
            min_score = 0
            max_score = float('inf')
            
            if start_time:
                min_score = start_time.timestamp()
            if end_time:
                max_score = end_time.timestamp()

            # Get videos within timestamp range
            timestamp_keys = self.text_client.zrangebyscore(
                "video_timestamps", min_score, max_score
            )

            for timestamp_key in timestamp_keys:
                # Extract key from timestamp key
                if isinstance(timestamp_key, bytes):
                    timestamp_key = timestamp_key.decode('utf-8')
                key = timestamp_key.replace(self.TIMESTAMP_PREFIX, "")
                metadata = self.retrieve_video(key, format="metadata")
                if metadata:
                    metadata["key"] = key
                    results.append(metadata)

            # Sort by timestamp (newest first)
            results.sort(key=lambda x: x.get("stored_at", ""), reverse=True)
            return results

        except redis.RedisError as e:
            self.logger.error(f"Redis error searching by timestamp range: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Error searching by timestamp range: {e}")
            return []

    def list_all_videos(self) -> List[Dict[str, Any]]:
        """
        List all stored videos with their metadata from Redis.

        Returns:
            List[Dict[str, Any]]: List of all video metadata
        """
        try:
            results = []
            all_keys = self.text_client.smembers("all_video_keys")
            
            for key in all_keys:
                metadata = self.retrieve_video(key, format="metadata")
                if metadata:
                    metadata["key"] = key
                    results.append(metadata)

            # Sort by timestamp (newest first)
            results.sort(key=lambda x: x.get("stored_at", ""), reverse=True)
            return results

        except redis.RedisError as e:
            self.logger.error(f"Redis error listing all videos: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Error listing all videos: {e}")
            return []

    def update_metadata(self, key: str, new_metadata: Dict[str, Any]) -> bool:
        """
        Update metadata for a stored video in Redis.

        Args:
            key: The key of the video to update
            new_metadata: New metadata to merge with existing metadata

        Returns:
            bool: True if update was successful, False otherwise
        """
        try:
            # Get existing metadata
            existing_metadata = self.retrieve_video(key, format="metadata")
            if existing_metadata is None:
                self.logger.warning(f"Video with key {key} not found for metadata update")
                return False

            # Update existing metadata
            existing_metadata.update(new_metadata)
            existing_metadata["updated_at"] = datetime.now().isoformat()

            # Update tags if provided
            if "tags" in new_metadata:
                old_tags = existing_metadata.get("tags", [])
                new_tags = new_metadata["tags"]

                # Remove old tag associations
                for tag in old_tags:
                    tag_key = f"{self.TAG_PREFIX}{tag}"
                    self.text_client.srem(tag_key, key)

                # Add new tag associations
                for tag in new_tags:
                    tag_key = f"{self.TAG_PREFIX}{tag}"
                    self.text_client.sadd(tag_key, key)

            # Store updated metadata
            metadata_key = f"{self.METADATA_PREFIX}{key}"
            self.text_client.set(metadata_key, json.dumps(existing_metadata))

            self.logger.info(f"Updated metadata for video {key}")
            return True

        except redis.RedisError as e:
            self.logger.error(f"Redis error updating metadata for video {key}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error updating metadata for video {key}: {e}")
            return False

    def delete_video(self, key: str) -> bool:
        """
        Delete a video and its associated data from Redis.

        Args:
            key: The key of the video to delete

        Returns:
            bool: True if deletion was successful, False otherwise
        """
        try:
            # Get metadata to check if video exists
            metadata = self.retrieve_video(key, format="metadata")
            if metadata is None:
                self.logger.warning(f"Video with key {key} not found for deletion")
                return False

            # Remove video data
            video_key = f"{self.VIDEO_PREFIX}{key}"
            self.redis_client.delete(video_key)

            # Remove metadata
            metadata_key = f"{self.METADATA_PREFIX}{key}"
            self.text_client.delete(metadata_key)

            # Remove tag associations
            tags = metadata.get("tags", [])
            for tag in tags:
                tag_key = f"{self.TAG_PREFIX}{tag}"
                self.text_client.srem(tag_key, key)

            # Remove from timestamp index
            timestamp_key = f"{self.TIMESTAMP_PREFIX}{key}"
            self.text_client.zrem("video_timestamps", timestamp_key)

            # Remove from global index
            self.text_client.srem("all_video_keys", key)

            self.logger.info(f"Deleted video {key}")
            return True

        except redis.RedisError as e:
            self.logger.error(f"Redis error deleting video {key}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error deleting video {key}: {e}")
            return False

    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics from Redis.

        Returns:
            Dict[str, Any]: Storage statistics including count, memory usage, and tag distribution
        """
        try:
            # Get total videos
            total_videos = self.text_client.scard("all_video_keys")
            
            # Get memory usage (approximate)
            info = self.redis_client.info("memory")
            used_memory_mb = info.get("used_memory", 0) / (1024 * 1024)
            
            # Count videos by tag
            tag_counts = {}
            all_keys = self.text_client.smembers("all_video_keys")
            
            for key in all_keys:
                metadata = self.retrieve_video(key, format="metadata")
                if metadata and "tags" in metadata:
                    for tag in metadata["tags"]:
                        tag_counts[tag] = tag_counts.get(tag, 0) + 1

            return {
                "total_videos": total_videos,
                "total_memory_mb": round(used_memory_mb, 2),
                "max_memory_mb": self.max_memory_mb,
                "memory_usage_percent": round((used_memory_mb / self.max_memory_mb) * 100, 2) if self.max_memory_mb > 0 else 0,
                "tag_distribution": tag_counts,
                "ttl_hours": self.ttl_hours,
                "redis_db": self.redis_client.connection_pool.connection_kwargs.get("db", 0)
            }

        except redis.RedisError as e:
            self.logger.error(f"Redis error getting storage stats: {e}")
            return {}
        except Exception as e:
            self.logger.error(f"Error getting storage stats: {e}")
            return {}

    def clear_all(self) -> bool:
        """
        Clear all stored videos and metadata from Redis.

        Returns:
            bool: True if operation was successful, False otherwise
        """
        try:
            # Get all keys
            all_keys = self.text_client.smembers("all_video_keys")
            
            # Delete each video
            for key in all_keys:
                self.delete_video(key)

            # Clear global indices
            self.text_client.delete("all_video_keys")
            self.text_client.delete("video_timestamps")

            self.logger.info("Cleared all stored videos and metadata")
            return True

        except redis.RedisError as e:
            self.logger.error(f"Redis error clearing all data: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error clearing all data: {e}")
            return False

    def __len__(self) -> int:
        """Return the number of stored videos."""
        try:
            return self.text_client.scard("all_video_keys")
        except redis.RedisError:
            return 0

    def __contains__(self, key: str) -> bool:
        """Check if a video with the given key exists."""
        try:
            return self.text_client.sismember("all_video_keys", key)
        except redis.RedisError:
            return False
