# Monocular Depth Estimation

This module provides a modular and testable implementation for monocular depth estimation using pre-trained models like Depth Anything v2.

## Architecture

The code has been refactored into a modular architecture with the following components:

### Core Components

1. **`DepthEstimator` class** (`depth_estimator.py`)
   - Main class that handles all depth estimation operations
   - Modular design with separate methods for each step
   - Supports multiple model backends
   - Comprehensive error handling

2. **Simple Interface** (`monocularDepthEstimation.py`)
   - Provides backward-compatible functions
   - Easy-to-use API for quick integration

3. **Unit Tests** (`test_depth_estimator.py`)
   - Comprehensive test coverage
   - Mocked dependencies for fast testing
   - Integration tests (optional)

## Usage

### Basic Usage

```python
from monocularDepthEstimation import estimate_depth_from_image

# Simple depth estimation
depth_map, original_image = estimate_depth_from_image("path/to/image.jpg")

# With visualization
depth_map, original_image = estimate_depth_from_image(
    "path/to/image.jpg",
    output_path="depth_result.png",
    show_result=True
)
```

### Advanced Usage

```python
from depth_estimator import DepthEstimator

# Create estimator with specific model
estimator = DepthEstimator("facebook/dpt-large")

# Load model explicitly
if estimator.load_model():
    # Preprocess image
    image = estimator.preprocess_image("path/to/image.jpg")

    # Estimate depth
    depth_map = estimator.estimate_depth(image)

    # Get statistics
    stats = estimator.get_depth_statistics(depth_map)
    print(f"Depth range: {stats['min_depth']} to {stats['max_depth']}")
```

## Testing

### Run Unit Tests

```bash
# Run all unit tests
python run_tests.py

# Run specific test file
python -m unittest test_depth_estimator.py -v

# Run with coverage (if coverage is installed)
coverage run -m unittest test_depth_estimator
coverage report
```

### Test Coverage

The test suite covers:

- ✅ Model initialization and loading
- ✅ Image preprocessing
- ✅ Depth estimation pipeline
- ✅ Depth map normalization
- ✅ Visualization creation
- ✅ Statistics calculation
- ✅ Error handling
- ✅ File I/O operations

## Dependencies

```bash
pip install torch torchvision transformers pillow opencv-python matplotlib
```

## Model Options

The system tries multiple models in order:

1. `LiheYoung/depth_anything_vitl14` (Depth Anything v2)
2. `facebook/dpt-large` (DPT Large)
3. `Intel/dpt-large` (Intel DPT)

## Key Features

### Modularity
- Separate concerns into different methods
- Easy to test individual components
- Clear separation between data processing and model inference

### Testability
- Comprehensive unit tests with mocking
- Integration tests for full pipeline
- Test coverage for error conditions

### Error Handling
- Graceful fallback to alternative models
- Clear error messages
- Proper exception handling

### Extensibility
- Easy to add new model backends
- Configurable visualization options
- Customizable preprocessing pipeline

## Performance Considerations

- Models are loaded once and reused
- Lazy loading of models (only when needed)
- Efficient memory usage with proper cleanup
- Support for GPU acceleration (if available)

## Future Enhancements

- Add support for video processing
- Implement batch processing
- Add more visualization options
- Support for custom model fine-tuning
- Real-time processing capabilities