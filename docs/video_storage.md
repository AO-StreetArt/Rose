# Video Storage System

The ROSE system now includes comprehensive video storage capabilities with support for both in-memory and Redis-based storage. This system allows you to store, retrieve, search, and manage video files with rich metadata and tag-based organization.

## Features

- **Flexible Storage Backends**: Choose between fast in-memory storage or persistent Redis storage
- **Rich Metadata Support**: Store comprehensive video information including duration, resolution, FPS, and custom fields
- **Tag-Based Organization**: Categorize videos with multiple tags for easy discovery
- **Timestamp Indexing**: Search videos by storage time or custom time ranges
- **Automatic Cleanup**: TTL-based expiration and memory management
- **Multiple Video Formats**: Support for MP4, WebM, AVI, MPEG, and other formats
- **Content Validation**: Automatic MIME type detection and size validation
- **Search Capabilities**: Find videos by tags, time ranges, or combinations of criteria

## Storage Classes

### MemoryVideoStorage

Fast in-memory storage for temporary video storage and processing.

```python
from rose.storage.memory_video_storage import MemoryVideoStorage

storage = MemoryVideoStorage(
    max_memory_mb=2048,      # Maximum memory usage
    ttl_hours=48,            # Default TTL for videos
    cleanup_interval=50,      # Operations before cleanup
    max_video_size_mb=100    # Maximum video file size
)
```

### RedisVideoStorage

Persistent Redis-based storage for scalable, distributed video storage.

```python
from rose.storage.redis_video_storage import RedisVideoStorage

storage = RedisVideoStorage(
    host="localhost",
    port=6379,
    db=0,
    password=None,
    max_memory_mb=2048,
    ttl_hours=48,
    max_video_size_mb=100
)
```

## Basic Usage

### Storing Videos

```python
# Create video metadata
metadata = {
    "title": "Sample Video",
    "description": "A demonstration video",
    "duration": 30.5,        # seconds
    "resolution": "1920x1080",
    "fps": 30.0,
    "source": "camera_1"
}

# Store video with tags
tags = ["demo", "sample", "camera"]
key = storage.store_video(
    video_data=video_bytes,
    metadata=metadata,
    tags=tags,
    filename="sample_video.mp4",
    ttl_hours=24
)
```

### Retrieving Videos

```python
# Get video data
video_data = storage.retrieve_video(key)

# Get metadata only
metadata = storage.retrieve_video(key, format="metadata")
```

### Searching Videos

```python
# Search by tags (OR operation)
demo_videos = storage.search_by_tags(["demo", "sample"])

# Search by tags (AND operation)
demo_and_short = storage.search_by_tags(["demo", "short"], operator="AND")

# Search by timestamp range
from datetime import datetime, timedelta

one_hour_ago = datetime.now() - timedelta(hours=1)
recent_videos = storage.search_by_timestamp_range(start_time=one_hour_ago)

# Search within specific time range
start_time = datetime(2024, 1, 1, 0, 0, 0)
end_time = datetime(2024, 1, 31, 23, 59, 59)
january_videos = storage.search_by_timestamp_range(start_time, end_time)
```

### Updating Metadata

```python
new_metadata = {
    "title": "Updated Title",
    "description": "New description",
    "tags": ["demo", "updated", "processed"]
}

success = storage.update_metadata(key, new_metadata)
```

### Deleting Videos

```python
# Delete specific video
success = storage.delete_video(key)

# Clear all videos
success = storage.clear_all()
```

## Metadata Schema

Videos are stored with the following metadata structure:

```python
{
    "title": "Video title",
    "description": "Video description",
    "duration": 30.5,                    # Duration in seconds
    "resolution": "1920x1080",           # Video resolution
    "fps": 30.0,                         # Frames per second
    "source": "camera_1",                # Source identifier
    "stored_at": "2024-01-15T10:30:00", # ISO timestamp
    "video_size_bytes": 1048576,         # File size in bytes
    "mime_type": "video/mp4",            # Detected MIME type
    "filename": "video.mp4",             # Original filename
    "tags": ["tag1", "tag2"],            # Associated tags
    "updated_at": "2024-01-15T11:00:00" # Last update timestamp
}
```

## Video Format Support

The system automatically detects and supports common video formats:

- **MP4**: `video/mp4`
- **WebM**: `video/webm`
- **AVI**: `video/x-msvideo`
- **MPEG**: `video/mpeg`
- **Other**: `video/octet-stream`

## Performance Considerations

### Memory Storage
- **Pros**: Fast access, no network latency, simple setup
- **Cons**: Limited by available RAM, data lost on restart
- **Best for**: Temporary storage, processing pipelines, development/testing

### Redis Storage
- **Pros**: Persistent, scalable, supports multiple clients
- **Cons**: Network latency, requires Redis server, memory usage
- **Best for**: Production systems, multi-service architectures, data persistence

## Error Handling

The storage classes include comprehensive error handling:

```python
try:
    key = storage.store_video(video_data, metadata, tags)
except ValueError as e:
    print(f"Validation error: {e}")
except redis.RedisError as e:
    print(f"Redis error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Configuration Options

### MemoryVideoStorage
- `max_memory_mb`: Maximum memory usage before cleanup
- `ttl_hours`: Default time-to-live for stored videos
- `cleanup_interval`: Operations before automatic cleanup
- `max_video_size_mb`: Maximum allowed video file size

### RedisVideoStorage
- `host`: Redis server hostname
- `port`: Redis server port
- `db`: Redis database number
- `password`: Redis authentication password
- `ttl_hours`: Default time-to-live for stored videos
- `max_video_size_mb`: Maximum allowed video file size

## Example Applications

### Video Processing Pipeline
```python
# Store incoming video
key = storage.store_video(raw_video, metadata, tags=["raw", "incoming"])

# Process video
processed_video = process_video(raw_video)
processed_metadata = metadata.copy()
processed_metadata["status"] = "processed"

# Store processed version
processed_key = storage.store_video(
    processed_video, 
    processed_metadata, 
    tags=["processed", "output"]
)

# Link videos
storage.update_metadata(key, {"processed_key": processed_key})
storage.update_metadata(processed_key, {"source_key": key})
```

### Video Archive System
```python
# Store with long TTL
archive_key = storage.store_video(
    video_data,
    metadata,
    tags=["archive", "long_term"],
    ttl_hours=8760  # 1 year
)

# Search archived videos
archived_videos = storage.search_by_tags(["archive"])
```

### Real-time Video Monitoring
```python
# Store with short TTL for real-time processing
monitoring_key = storage.store_video(
    video_data,
    metadata,
    tags=["monitoring", "real_time"],
    ttl_hours=1  # 1 hour
)

# Get recent videos
recent_videos = storage.search_by_timestamp_range(
    start_time=datetime.now() - timedelta(minutes=30)
)
```

## Running the Demo

To see the video storage system in action, run the demo script:

```bash
cd examples
python video_storage_demo.py
```

This will demonstrate all the major features of both storage backends.

## Dependencies

- **Memory Storage**: No external dependencies
- **Redis Storage**: Requires `redis` Python package and Redis server
- **Common**: Standard library modules (`hashlib`, `datetime`, `logging`, etc.)

## Future Enhancements

- Video thumbnail generation
- Video compression and format conversion
- Streaming support for large videos
- Integration with cloud storage providers
- Advanced search and filtering options
- Video analytics and metrics collection
