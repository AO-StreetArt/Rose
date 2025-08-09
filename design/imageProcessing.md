# Image Processing

## Algorithms

Below is a review of image processing techniques currently implemented.

### Depth Estimation

Identify how far a given pixel is from the camera.  
Output - Depth Map

### Edge Detection

Find edges of objects within the image.  These are boundaries within the image.
Output - Edge Map

### Feature Detector

Finds keypoints within the image.  These are major landmarks, typically corners, that can be aligned between images taken at different angles.
Output - Keypoint listing with coordinates & descriptors

### Image Segmenter

Identifies different areas within the image.
Output - Segmented Mask

### Image to 3D

Takes a simple image of a single object and converts it into a 3D model.  The heaviest of the bunch, and also the hardest to use, this algorithm + depth estimation is what really takes us from 2D to 3D fully within a scene, at least for monocular vision.  The importance of this goes down with binocular vision, as a lot of 3D detail can be taken from the differences in the two eyes.
Output - a 3D model

### Object Detector

What's in the image?  What do we recognize?  Cars, people, chickens, etc?  

In a final system, re-training this constantly based on new information will have a HUGE impact.  This lets us recognize new things, which is part of learning.

Output - Different object classifications (string tags), with confidence values

### Video Classifier

What's in the video?  What do we recognize?

This is async in the background, and runs on sections of video taken from the camera every so often.

Output - Different video classifications (string tags), with confidence values

### Feature Extractor

Not used directly in final pipeline, used by Image Comparator

### Image Comparator

How similar are two images?  Relies on Feature Extraction to identify important elements to compare.

## Putting it All Together

Ok, so fundamentally, we're getting one frame at a time from a camera.  So we have a 'current image', a buffer of 'next images', and 'previous images'.  

There's an order in which we need to do things - the 'current image' needs to flow through a pipeline, get placed in 'previous images', and at each step in the process we're emitting new information to the decision-making cortex.  Things at the front of the pipeline are for supporting basic functions like movement.  Then, we get into higher-level thinking in the rear end of the 'current image' pipeline, and in processing the 'previous images' where we look at things like inter-frame analysis and video analysis.  

I will note that the "right" answer for the ordering here (and whether we use binocular vs monocular depth detection), is going to be something that can change based on the machine in question.  In nature, binocular vision is more common in predators for whom accurate depth perception is critical to catching prey, while monocular vision is more common in prey animals who prioritize a wide field of vision.  We're going to focus on monocular here.

I'm also honestly torn on how much we want to do in terms of inter-frame analysis vs invest time in models designed for 3D.  As these are less common, though, we should focus in on inter-frame analysis in the short term.

### Current Image Pipeline

1. Depth Estimation
2. Edge Detection
3. Object Detection (w/ async recall from memory)
4. Image Segmentation
5. Feature Detection (only really necessary for binocular depth perception)

### Inter-Frame Analysis

1. Image Comparison (specific segments of overall image)
2. Object tracking & velocity calculations

### Previous Video Pipeline

1. Video Classification

### Memory & Retention

Now, we need to store all this data, along with analysis from the result