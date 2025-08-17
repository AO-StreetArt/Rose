# Velocity Tracking in Video Stream Processing

This document describes the velocity tracking functionality added to the `process_video_stream.py` script.

## Overview

The velocity tracking system analyzes video frames to detect objects, track their movement across frames, and calculate their 3D velocity using depth estimation and image segmentation.

## Features

- **Object Detection**: Uses existing object detection models (Faster R-CNN, SSD)
- **Depth Estimation**: Integrates with DPT and ZoeDepth models
- **Image Segmentation**: Uses detected objects as prompts for segmentation
- **Object Tracking**: Tracks objects across frames using image similarity
- **Velocity Calculation**: Calculates 3D velocity using position and depth changes
- **Storage Management**: Uses memory storage for real-time processing and Redis for persistence

## How It Works

### 1. Frame Processing Pipeline

1. **Initial Analysis**: Each frame is processed for depth estimation, object detection, and image segmentation
2. **Parallel Processing**: The initial analysis runs in parallel for optimal performance
3. **Velocity Tracking**: After initial analysis, results are sent to a separate thread for velocity calculation
4. **Non-blocking**: Velocity tracking doesn't block the main video display

### 2. Object Tracking Algorithm

1. **Initial Population**: If memory storage is empty, all detected objects are stored with metadata
2. **Image Comparison**: For each new detection, the system searches for similar objects in storage
3. **Similarity Matching**: Uses VGG16 features to compare object regions and find matches
4. **Match ID Generation**: When a match is found, a unique UUID is generated to link the objects
5. **Velocity Calculation**: 3D velocity is calculated using position and depth changes over time

### 3. Storage Strategy

1. **Memory Storage**: Fast, in-memory storage for real-time processing
2. **Redis Sync**: Periodically syncs data to Redis for persistence
3. **Memory Cleanup**: Clears memory storage after Redis sync to prevent memory bloat
4. **Metadata Management**: Stores bounding boxes, depth estimates, timestamps, and velocity data

## Usage

### Basic Usage

```bash
# Run with default settings (includes Redis)
python process_video_stream.py

# Run without Redis (memory-only mode)
python process_video_stream.py --no-redis

# Customize object detection
python process_video_stream.py --object-confidence 0.7 --object-model ssd

# Adjust processing frequency
python process_video_stream.py --frame-skip 2 --fps 60
```

### Command Line Options

- `--no-redis`: Disable Redis storage and use memory-only mode
- `--object-confidence`: Set minimum confidence for object detection
- `--object-model`: Choose between 'faster_rcnn' and 'ssd'
- `--frame-skip`: Process every Nth frame (higher values = faster processing)
- `--fps`: Target frames per second
- `--camera`: Camera device ID

## Technical Details

### Velocity Calculation

The 3D velocity is calculated using:

```
velocity = √(dx² + dy² + dz²) / Δt
```

Where:
- `dx, dy`: Pixel coordinate differences converted to meters
- `dz`: Depth difference in meters
- `Δt`: Time difference between frames

### Image Similarity

- **Method**: VGG16 feature extraction with cosine similarity
- **Threshold**: 0.6 (configurable)
- **Normalization**: Features are normalized before comparison
- **Resizing**: Images are resized to match dimensions for comparison

### Storage Metadata

Each stored object includes:
- Bounding box coordinates
- Object class name and confidence
- Depth estimate
- Center point coordinates
- Timestamp and frame number
- Match ID (if matched with previous detection)
- Velocity in meters per second
- Previous detection key (for tracking)

## Performance Considerations

- **Memory Usage**: Memory storage is limited to 512MB by default
- **Processing Speed**: Velocity tracking runs on separate threads to avoid blocking
- **Storage Cleanup**: Automatic cleanup prevents memory bloat
- **Redis Fallback**: Gracefully falls back to memory-only mode if Redis is unavailable

## Error Handling

- **Graceful Degradation**: Continues processing even if individual detections fail
- **Exception Logging**: Detailed error logging with stack traces
- **Connection Recovery**: Handles Redis connection failures gracefully
- **Image Processing**: Continues processing even if image comparison fails

## Testing

### Running Tests

The velocity tracking system includes comprehensive unit tests and integration tests.

#### Unit Tests

Run the dedicated unit test suite:

```bash
# Run unit tests only
python test_velocity_tracking_unit.py

# Run with test runner
python run_velocity_tests.py
```

#### Integration Tests

Run the integration tests that verify the complete system:

```bash
# Run integration tests
python test_velocity_tracking.py
```

#### Test Coverage

The unit tests cover:

- **Object Detection Storage**: Testing initial population of detections
- **Depth Calculation**: Various scenarios including edge cases and invalid inputs
- **Image Matching**: Finding best matches using similarity thresholds
- **Velocity Calculation**: 3D velocity computation with time validation
- **Metadata Management**: Updating and storing object metadata
- **Error Handling**: Graceful handling of exceptions and edge cases
- **Redis Integration**: Testing storage synchronization and failure handling

#### Test Scenarios

- Empty storage initialization
- Invalid bounding box handling
- Out-of-bounds coordinate handling
- Time difference edge cases (zero, negative, very small)
- Image dimension mismatches
- Redis connection failures
- Individual detection processing errors

### Manual Testing

Test the basic functionality without requiring a camera:

```bash
# Test storage systems
python test_velocity_tracking.py

# Test image comparison
python -c "
from rose.processing.image_comparator import ImageComparator
import numpy as np

comparator = ImageComparator()
img1 = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
img2 = img1.copy() + np.random.randint(-20, 20, (64, 64, 3), dtype=np.uint8)
img2 = np.clip(img2, 0, 255).astype(np.uint8)

result = comparator.compare_images(img1, img2, method='vgg16', normalize=True)
print(f'Similarity: {result[\"similarity_score\"]:.3f}')
"
```

## Dependencies

- `rose.storage.memory_image_storage`: In-memory image storage
- `rose.storage.redis_image_storage`: Redis-based image storage
- `rose.processing.image_comparator`: Image similarity comparison
- `rose.processing.depth_estimator`: Depth estimation
- `rose.processing.object_detector`: Object detection
- `rose.processing.image_segmenter`: Image segmentation

## Future Enhancements

- **Kalman Filtering**: Improve tracking accuracy with motion prediction
- **Multi-object Tracking**: Handle occlusions and object interactions
- **Trajectory Analysis**: Analyze object movement patterns over time
- **Performance Optimization**: GPU acceleration for image comparison
- **Configurable Thresholds**: User-adjustable similarity and velocity thresholds
