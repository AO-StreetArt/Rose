# 3D Processing

Monocular 3D Processing is built on two algorithms specifically:

* Monocular Depth detection
* Image to 3D Algorithms

In addition, we will rely heavily on the outputs of other image processing algorithms, in order to run the 'image to 3d' algorithm on specific parts of an image, and measure the depth of different objects.

A few notes:

* Binocular 3D reconstruction follows a similar pipeline, but relies on very different implementations for every single step.
* It's probably not worth running the full 3D processing pipeline on each frame of a video.  Instead, we should try to identify key frames and do additional processing on those.

### Object Detection & Depth Mapping

To begin with, we start by using the outputs of the Image Processing pipeline to separate out major components of the image.  Each of these is defined by segments and outlines, and we can take the average depth of the pixels within that segment as the depth of each object.

So now, we have a 3D mapping - using the camera as the origin of the coordinate system, we can plot the placement of each unique object in the image in 3D space.

### Object to 3D

Once we've identified the different objects in the image, we can separate the image into different pieces, and run them individually through an image-to-3D algorithm.  These models are tuned to work on simple images, so we really want to also generate a mask for them to simplify down to literally *only* the parts we want the model to process.