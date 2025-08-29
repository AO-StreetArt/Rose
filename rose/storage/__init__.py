"""
Storage module for the ROSE system.

This module provides storage abstractions for images and videos with support for:
- Memory-based storage for fast access
- Redis-based storage for persistence and scalability
- Tag-based indexing and search
- Metadata management
- TTL and cleanup functionality
"""

from .memory_image_storage import MemoryImageStorage
from .redis_image_storage import RedisImageStorage
from .memory_video_storage import MemoryVideoStorage
from .redis_video_storage import RedisVideoStorage

__all__ = [
    "MemoryImageStorage",
    "RedisImageStorage", 
    "MemoryVideoStorage",
    "RedisVideoStorage"
]
