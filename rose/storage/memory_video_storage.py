#!/usr/bin/env python3
"""
Memory-based video storage system for storing video files by timestamp and tags.

This module provides functionality to:
- Store video files with metadata and tags in memory
- Retrieve videos by tags, keys, or timestamp ranges
- Manage video lifecycle and cleanup
- Support various video formats and metadata
"""

import hashlib
import os
from typing import Dict, List, Optional, Union, Any, BinaryIO
from datetime import datetime, timedelta
import logging
from collections import defaultdict
import mimetypes


class MemoryVideoStorage:
    """
    In-memory storage system for video files and metadata.

    This class provides functionality to:
    - Store video files with metadata and tags in memory
    - Retrieve videos by tags, keys, or timestamp ranges
    - Manage video lifecycle and cleanup
    - Support various video formats
    - Compatible with RedisVideoStorage for seamless data transfer
    """

    def __init__(
        self,
        max_memory_mb: int = 2048,
        ttl_hours: int = 48,
        cleanup_interval: int = 50,
        max_video_size_mb: int = 100
    ):
        """
        Initialize in-memory video storage.

        Args:
            max_memory_mb: Maximum memory usage in MB before cleanup
            ttl_hours: Default TTL for stored videos in hours
            cleanup_interval: Number of operations before automatic cleanup
            max_video_size_mb: Maximum video file size in MB
        """
        self.max_memory_mb = max_memory_mb
        self.ttl_hours = ttl_hours
        self.cleanup_interval = cleanup_interval
        self.max_video_size_mb = max_video_size_mb
        self.logger = logging.getLogger(__name__)
        self.operation_count = 0

        # In-memory storage structures
        self.videos: Dict[str, bytes] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.tags: Dict[str, set] = defaultdict(set)
        self.key_to_tags: Dict[str, List[str]] = {}
        self.expiry_times: Dict[str, datetime] = {}
        self.all_keys: set = set()
        self.timestamp_index: Dict[datetime, List[str]] = defaultdict(list)

        # Key prefixes for compatibility with RedisVideoStorage
        self.VIDEO_PREFIX = "vid:"
        self.METADATA_PREFIX = "meta:"
        self.TAG_PREFIX = "tag:"
        self.INDEX_PREFIX = "idx:"
        self.TIMESTAMP_PREFIX = "ts:"

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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # Include microseconds
        
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

    def _check_memory_limit(self) -> None:
        """Check if memory usage exceeds limit and cleanup if necessary."""
        current_memory_mb = sum(len(vid) for vid in self.videos.values()) / (1024 * 1024)

        if current_memory_mb > self.max_memory_mb:
            self.logger.warning(f"Memory limit exceeded ({current_memory_mb:.2f}MB), cleaning up...")
            self._cleanup_expired()

    def _cleanup_expired(self) -> None:
        """Remove expired videos and metadata."""
        current_time = datetime.now()
        expired_keys = []

        for key, expiry_time in self.expiry_times.items():
            if current_time > expiry_time:
                expired_keys.append(key)

        for key in expired_keys:
            self.delete_video(key)

        if expired_keys:
            self.logger.info(f"Cleaned up {len(expired_keys)} expired videos")

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
        Store a video with metadata and tags.

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
        """
        try:
            # Validate and convert video data
            video_bytes = self._validate_video_data(video_data)

            # Generate key if not provided
            if key is None:
                key = self._generate_video_key(video_bytes, filename, self.VIDEO_PREFIX)

            # Store video data
            self.videos[key] = video_bytes

            # Detect MIME type
            mime_type = self._detect_mime_type(video_bytes, filename)

            # Store metadata
            metadata["stored_at"] = datetime.now().isoformat()
            metadata["video_size_bytes"] = len(video_bytes)
            metadata["mime_type"] = mime_type
            metadata["filename"] = filename
            metadata["tags"] = tags or []
            metadata["duration"] = metadata.get("duration", 0)  # Duration in seconds
            metadata["resolution"] = metadata.get("resolution", "unknown")
            metadata["fps"] = metadata.get("fps", 0)

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

            # Index by timestamp
            stored_time = datetime.fromisoformat(metadata["stored_at"])
            self.timestamp_index[stored_time].append(key)

            # Add to global index
            self.all_keys.add(key)

            # Increment operation count and check memory
            self.operation_count += 1
            if self.operation_count % self.cleanup_interval == 0:
                self._check_memory_limit()

            # Also check memory limit if we're close to the limit
            current_memory_mb = len(video_bytes) / (1024 * 1024)
            if current_memory_mb > self.max_memory_mb * 0.8:  # Check at 80% of limit
                self._check_memory_limit()

            self.logger.info(f"Stored video with key: {key}, size: {len(video_bytes)} bytes")
            return key

        except Exception as e:
            self.logger.error(f"Error storing video: {e}")
            raise

    def retrieve_video(self, key: str, format: str = "bytes") -> Optional[Union[bytes, Dict[str, Any]]]:
        """
        Retrieve a video by its key.

        Args:
            key: The key of the video to retrieve
            format: Return format ("bytes" for raw video data, "metadata" for metadata only)

        Returns:
            Optional[Union[bytes, Dict[str, Any]]]: Video data or metadata, or None if not found
        """
        try:
            if key not in self.videos:
                self.logger.warning(f"Video with key {key} not found")
                return None

            # Check if expired
            if key in self.expiry_times and datetime.now() > self.expiry_times[key]:
                self.logger.warning(f"Video with key {key} has expired")
                self.delete_video(key)
                return None

            if format == "metadata":
                return self.metadata.get(key, {})
            else:
                return self.videos[key]

        except Exception as e:
            self.logger.error(f"Error retrieving video {key}: {e}")
            return None

    def search_by_tags(self, tags: List[str], operator: str = "OR") -> List[Dict[str, Any]]:
        """
        Search for videos by tags.

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
                    matching_keys = self.tags.get(tags[0], set()).copy()
                    for tag in tags[1:]:
                        matching_keys &= self.tags.get(tag, set())
            else:
                # OR: Any tag can match
                for tag in tags:
                    matching_keys.update(self.tags.get(tag, set()))

            results = []
            for key in matching_keys:
                if key in self.metadata:
                    metadata = self.metadata[key].copy()
                    metadata["key"] = key
                    results.append(metadata)

            return results

        except Exception as e:
            self.logger.error(f"Error searching by tags: {e}")
            return []

    def search_by_timestamp_range(
        self, 
        start_time: Optional[datetime] = None, 
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for videos within a timestamp range.

        Args:
            start_time: Start time for the range (inclusive)
            end_time: End time for the range (inclusive)

        Returns:
            List[Dict[str, Any]]: List of video metadata within the timestamp range
        """
        try:
            results = []
            
            for timestamp, keys in self.timestamp_index.items():
                # Check if timestamp is within range
                if start_time and timestamp < start_time:
                    continue
                if end_time and timestamp > end_time:
                    continue
                
                # Add videos from this timestamp
                for key in keys:
                    if key in self.metadata:
                        metadata = self.metadata[key].copy()
                        metadata["key"] = key
                        results.append(metadata)

            # Sort by timestamp (newest first)
            results.sort(key=lambda x: x.get("stored_at", ""), reverse=True)
            return results

        except Exception as e:
            self.logger.error(f"Error searching by timestamp range: {e}")
            return []

    def list_all_videos(self) -> List[Dict[str, Any]]:
        """
        List all stored videos with their metadata.

        Returns:
            List[Dict[str, Any]]: List of all video metadata
        """
        try:
            results = []
            for key in self.all_keys:
                if key in self.metadata:
                    metadata = self.metadata[key].copy()
                    metadata["key"] = key
                    results.append(metadata)

            # Sort by timestamp (newest first)
            results.sort(key=lambda x: x.get("stored_at", ""), reverse=True)
            return results

        except Exception as e:
            self.logger.error(f"Error listing all videos: {e}")
            return []

    def update_metadata(self, key: str, new_metadata: Dict[str, Any]) -> bool:
        """
        Update metadata for a stored video.

        Args:
            key: The key of the video to update
            new_metadata: New metadata to merge with existing metadata

        Returns:
            bool: True if update was successful, False otherwise
        """
        try:
            if key not in self.metadata:
                self.logger.warning(f"Video with key {key} not found for metadata update")
                return False

            # Update existing metadata
            self.metadata[key].update(new_metadata)
            self.metadata[key]["updated_at"] = datetime.now().isoformat()

            # Update tags if provided
            if "tags" in new_metadata:
                old_tags = self.key_to_tags.get(key, [])
                new_tags = new_metadata["tags"]

                # Remove old tag associations
                for tag in old_tags:
                    if key in self.tags[tag]:
                        self.tags[tag].remove(key)

                # Add new tag associations
                self.key_to_tags[key] = new_tags.copy()
                for tag in new_tags:
                    self.tags[tag].add(key)

            self.logger.info(f"Updated metadata for video {key}")
            return True

        except Exception as e:
            self.logger.error(f"Error updating metadata for video {key}: {e}")
            return False

    def delete_video(self, key: str) -> bool:
        """
        Delete a video and its associated data.

        Args:
            key: The key of the video to delete

        Returns:
            bool: True if deletion was successful, False otherwise
        """
        try:
            if key not in self.videos:
                self.logger.warning(f"Video with key {key} not found for deletion")
                return False

            # Remove from all storage structures
            video_data = self.videos.pop(key, None)
            metadata = self.metadata.pop(key, None)
            tags = self.key_to_tags.pop(key, [])
            expiry_time = self.expiry_times.pop(key, None)
            self.all_keys.discard(key)

            # Remove tag associations
            for tag in tags:
                if key in self.tags[tag]:
                    self.tags[tag].remove(key)

            # Remove from timestamp index
            if metadata and "stored_at" in metadata:
                stored_time = datetime.fromisoformat(metadata["stored_at"])
                if stored_time in self.timestamp_index and key in self.timestamp_index[stored_time]:
                    self.timestamp_index[stored_time].remove(key)
                    if not self.timestamp_index[stored_time]:
                        del self.timestamp_index[stored_time]

            if video_data:
                freed_memory_mb = len(video_data) / (1024 * 1024)
                self.logger.info(f"Deleted video {key}, freed {freed_memory_mb:.2f}MB")

            return True

        except Exception as e:
            self.logger.error(f"Error deleting video {key}: {e}")
            return False

    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.

        Returns:
            Dict[str, Any]: Storage statistics including count, memory usage, and tag distribution
        """
        try:
            total_videos = len(self.videos)
            total_memory_mb = sum(len(vid) for vid in self.videos.values()) / (1024 * 1024)
            
            # Count videos by tag
            tag_counts = {tag: len(keys) for tag, keys in self.tags.items()}
            
            # Count expired videos
            expired_count = sum(1 for expiry_time in self.expiry_times.values() 
                              if datetime.now() > expiry_time)

            return {
                "total_videos": total_videos,
                "total_memory_mb": round(total_memory_mb, 2),
                "max_memory_mb": self.max_memory_mb,
                "memory_usage_percent": round((total_memory_mb / self.max_memory_mb) * 100, 2),
                "tag_distribution": tag_counts,
                "expired_videos": expired_count,
                "ttl_hours": self.ttl_hours,
                "operation_count": self.operation_count
            }

        except Exception as e:
            self.logger.error(f"Error getting storage stats: {e}")
            return {}

    def clear_all(self) -> bool:
        """
        Clear all stored videos and metadata.

        Returns:
            bool: True if operation was successful, False otherwise
        """
        try:
            self.videos.clear()
            self.metadata.clear()
            self.tags.clear()
            self.key_to_tags.clear()
            self.expiry_times.clear()
            self.all_keys.clear()
            self.timestamp_index.clear()
            self.operation_count = 0

            self.logger.info("Cleared all stored videos and metadata")
            return True

        except Exception as e:
            self.logger.error(f"Error clearing all data: {e}")
            return False

    def __len__(self) -> int:
        """Return the number of stored videos."""
        return len(self.videos)

    def __contains__(self, key: str) -> bool:
        """Check if a video with the given key exists."""
        return key in self.videos
