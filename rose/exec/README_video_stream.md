# Video Stream Processing

This module provides real-time video processing capabilities using Rose modules. It opens a webcam connection and processes each frame with depth estimation, object detection, and image segmentation, displaying results in separate windows.

## Features

- **Real-time webcam processing**: Opens camera connection and processes frames continuously
- **Depth estimation**: Uses DPT or ZoeDepth models for monocular depth estimation
- **Object detection**: Detects objects using Faster R-CNN or SSD models
- **Image segmentation**: Uses detected objects as text prompts for CLIPSeg segmentation
- **Multi-window display**: Shows original frame with detections, depth map, and segmentation results
- **Threaded processing**: Uses background threads to maintain real-time performance
- **Configurable parameters**: Adjustable camera settings, model parameters, and visualization options

## Usage

### Basic Usage

```bash
# Process video stream with default settings
python process_video_stream.py

# Use a different camera
python process_video_stream.py --camera 1

# Adjust frame rate
python process_video_stream.py --fps 15
```

### Advanced Usage

```bash
# Use ZoeDepth model for depth estimation
python process_video_stream.py --zoedepth

# Adjust object detection confidence
python process_video_stream.py --object-confidence 0.7

# Use SSD model for object detection
python process_video_stream.py --object-model ssd

# Change depth visualization colormap
python process_video_stream.py --colormap plasma

# Limit number of objects for segmentation
python process_video_stream.py --max-objects 3
```

### Command Line Options

- `--camera`: Camera device ID (default: 0)
- `--fps`: Target frames per second (default: 30)
- `--zoedepth`: Use ZoeDepth model instead of DPT for depth estimation
- `--object-confidence`: Minimum confidence threshold for object detection (default: 0.5)
- `--object-model`: Object detection model to use (`faster_rcnn` or `ssd`, default: `faster_rcnn`)
- `--colormap`: Colormap for depth visualization (`viridis`, `plasma`, `inferno`, `magma`, `cividis`, `gray`, default: `viridis`)
- `--max-objects`: Maximum number of objects to use for segmentation prompts (default: 5)

## Controls

- **'q'**: Quit the application
- **'s'**: Save current frame and depth map to files

## Architecture

### VideoProcessor Class

The `VideoProcessor` class handles the core processing functionality:

- **Initialization**: Sets up depth estimator, object detector, and image segmenter
- **Frame processing**: Processes individual frames through the complete pipeline
- **Threading**: Uses background threads to maintain real-time performance
- **Visualization**: Creates visualizations for depth maps and segmentation results

### Processing Pipeline

1. **Frame capture**: Read frame from webcam
2. **Depth estimation**: Generate depth map using DPT or ZoeDepth
3. **Object detection**: Detect objects in the frame
4. **Prompt extraction**: Extract object class names for segmentation
5. **Image segmentation**: Perform segmentation using detected objects as prompts
6. **Visualization**: Create visualizations for display
7. **Display**: Show results in separate windows

### Threading Model

- **Main thread**: Handles frame capture, display, and user input
- **Processing thread**: Runs the AI models in the background
- **Queue system**: Manages frame flow between threads with size limits

## Performance Considerations

- **Model loading**: Models are loaded once during initialization
- **Queue management**: Frame queues prevent memory overflow
- **Resolution**: Default 640x480 resolution balances performance and quality
- **Frame rate**: Adjustable FPS to match processing capabilities
- **GPU acceleration**: Automatic GPU detection for supported models

## Dependencies

- OpenCV (cv2) for video capture and display
- PyTorch for deep learning models
- Transformers for pre-trained models
- PIL for image processing
- NumPy for numerical operations
- Matplotlib for colormap visualization

## Troubleshooting

### Camera Issues
- Ensure camera is not in use by another application
- Try different camera IDs (0, 1, 2, etc.)
- Check camera permissions

### Performance Issues
- Reduce frame rate with `--fps` option
- Lower object detection confidence threshold
- Use DPT instead of ZoeDepth for faster processing
- Reduce maximum objects for segmentation

### Model Loading Issues
- Ensure internet connection for model downloads
- Check available disk space for model caching
- Verify PyTorch installation

## Examples

### Basic Real-time Processing
```bash
python process_video_stream.py
```

### High-quality Processing
```bash
python process_video_stream.py --zoedepth --object-confidence 0.8 --colormap inferno
```

### Fast Processing
```bash
python process_video_stream.py --fps 15 --object-model ssd --max-objects 2
```

## Integration

The video processing functionality can be integrated into larger applications by:

1. Importing the `VideoProcessor` class
2. Creating an instance with desired parameters
3. Processing frames individually or in batches
4. Using the visualization methods for custom displays

```python
from rose.exec.process_video_stream import VideoProcessor

processor = VideoProcessor(use_zoedepth=True, object_confidence=0.6)
result = processor.process_frame(frame)
original, depth, seg = processor.create_visualization(result)
``` 