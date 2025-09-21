#!/usr/bin/env python3
"""
Demo script showing how to use the video storage classes.

This script demonstrates:
- Storing videos with metadata and tags
- Searching videos by tags and timestamp ranges
- Updating metadata
- Deleting videos
- Getting storage statistics
"""

import os
import sys
from pathlib import Path
import tempfile
import time
from datetime import datetime, timedelta

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rose.storage.memory_video_storage import MemoryVideoStorage
from rose.storage.redis_video_storage import RedisVideoStorage


def create_sample_video_data(size_mb: float = 1.0) -> bytes:
    """Create sample video data for demonstration purposes."""
    # Create a simple video-like header followed by random data
    header = b'\x00\x00\x00\x18ftypmp4'  # MP4 header
    random_data = os.urandom(int(size_mb * 1024 * 1024 - len(header)))
    return header + random_data


def demo_memory_video_storage():
    """Demonstrate memory-based video storage functionality."""
    print("\n=== Memory Video Storage Demo ===")
    
    # Initialize storage
    storage = MemoryVideoStorage(max_memory_mb=100, ttl_hours=1)
    
    # Create sample video data
    video_data = create_sample_video_data(0.5)  # 0.5MB
    
    # Store a video with metadata and tags
    metadata = {
        "title": "Sample Video 1",
        "description": "A demonstration video",
        "duration": 30.5,  # seconds
        "resolution": "1920x1080",
        "fps": 30.0,
        "source": "demo"
    }
    
    tags = ["demo", "sample", "test", "video"]
    
    print("Storing video...")
    key = storage.store_video(
        video_data=video_data,
        metadata=metadata,
        tags=tags,
        filename="sample_video_1.mp4"
    )
    print(f"Stored video with key: {key}")
    
    # Store another video
    video_data2 = create_sample_video_data(0.3)  # 0.3MB
    metadata2 = {
        "title": "Sample Video 2",
        "description": "Another demonstration video",
        "duration": 15.2,
        "resolution": "1280x720",
        "fps": 25.0,
        "source": "demo"
    }
    
    tags2 = ["demo", "sample", "short"]
    
    key2 = storage.store_video(
        video_data=video_data2,
        metadata=metadata2,
        tags=tags2,
        filename="sample_video_2.mp4"
    )
    print(f"Stored second video with key: {key2}")
    
    # List all videos
    print("\nAll stored videos:")
    all_videos = storage.list_all_videos()
    for video in all_videos:
        print(f"  - {video['title']} (key: {video['key']})")
        print(f"    Tags: {video['tags']}")
        print(f"    Size: {video['video_size_bytes'] / 1024:.1f} KB")
    
    # Search by tags
    print("\nSearching for videos with tag 'demo':")
    demo_videos = storage.search_by_tags(["demo"])
    for video in demo_videos:
        print(f"  - {video['title']}")
    
    print("\nSearching for videos with tags 'demo' AND 'short':")
    short_demo_videos = storage.search_by_tags(["demo", "short"], operator="AND")
    for video in short_demo_videos:
        print(f"  - {video['title']}")
    
    # Search by timestamp range
    print("\nSearching for videos in the last hour:")
    one_hour_ago = datetime.now() - timedelta(hours=1)
    recent_videos = storage.search_by_timestamp_range(start_time=one_hour_ago)
    for video in recent_videos:
        print(f"  - {video['title']} (stored at: {video['stored_at']})")
    
    # Update metadata
    print("\nUpdating metadata for first video...")
    new_metadata = {
        "title": "Updated Sample Video 1",
        "description": "Updated description",
        "tags": ["demo", "sample", "test", "video", "updated"]
    }
    success = storage.update_metadata(key, new_metadata)
    print(f"Metadata update successful: {success}")
    
    # Retrieve video metadata
    print(f"\nRetrieved metadata for {key}:")
    retrieved_metadata = storage.retrieve_video(key, format="metadata")
    if retrieved_metadata:
        print(f"  Title: {retrieved_metadata['title']}")
        print(f"  Description: {retrieved_metadata['description']}")
        print(f"  Tags: {retrieved_metadata['tags']}")
        print(f"  Updated at: {retrieved_metadata.get('updated_at', 'Not updated')}")
    
    # Get storage statistics
    print("\nStorage statistics:")
    stats = storage.get_storage_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Delete a video
    print(f"\nDeleting video {key2}...")
    success = storage.delete_video(key2)
    print(f"Deletion successful: {success}")
    
    print(f"Remaining videos: {len(storage)}")
    
    # Clear all
    print("\nClearing all videos...")
    success = storage.clear_all()
    print(f"Clear all successful: {success}")
    print(f"Remaining videos: {len(storage)}")


def demo_redis_video_storage():
    """Demonstrate Redis-based video storage functionality."""
    print("\n=== Redis Video Storage Demo ===")
    
    try:
        # Initialize storage (will fail if Redis is not running)
        storage = RedisVideoStorage(
            host="localhost",
            port=6379,
            db=1,  # Use different DB to avoid conflicts
            ttl_hours=1
        )
        
        # Create sample video data
        video_data = create_sample_video_data(0.2)  # 0.2MB
        
        # Store a video
        metadata = {
            "title": "Redis Sample Video",
            "description": "A video stored in Redis",
            "duration": 20.0,
            "resolution": "1280x720",
            "fps": 24.0,
            "source": "redis_demo"
        }
        
        tags = ["redis", "demo", "persistent"]
        
        print("Storing video in Redis...")
        key = storage.store_video(
            video_data=video_data,
            metadata=metadata,
            tags=tags,
            filename="redis_sample.mp4"
        )
        print(f"Stored video with key: {key}")
        
        # List all videos
        print("\nAll videos in Redis:")
        all_videos = storage.list_all_videos()
        for video in all_videos:
            print(f"  - {video['title']} (key: {video['key']})")
        
        # Search by tags
        print("\nSearching for videos with tag 'redis':")
        redis_videos = storage.search_by_tags(["redis"])
        for video in redis_videos:
            print(f"  - {video['title']}")
        
        # Get storage statistics
        print("\nRedis storage statistics:")
        stats = storage.get_storage_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Clear all
        print("\nClearing all videos from Redis...")
        success = storage.clear_all()
        print(f"Clear all successful: {success}")
        
    except Exception as e:
        print(f"Redis demo failed (Redis may not be running): {e}")
        print("To run this demo, start Redis server first")


def main():
    """Run the video storage demos."""
    print("Video Storage System Demo")
    print("=" * 50)
    
    # Memory storage demo
    demo_memory_video_storage()
    
    # Redis storage demo (will fail gracefully if Redis is not running)
    demo_redis_video_storage()
    
    print("\n" + "=" * 50)
    print("Demo completed!")


if __name__ == "__main__":
    main()
