# Generate a 3D Scene from a Picture

Our objective is to take a 2d picture of a real place, and generate a 3d scene from it.

So, we start with object detection, image segmentation, and depth estimation, much like our process_video_stream script

Then, however, we are going to do a few things:

1. For each object detected:
    a. Generate a corresponding mask based on image segmentation
    b. Create a new temp image, which is just the detected object, with the mask applied and clipped to the object's bounding box
    c. Execute 2d to 3d algorithm on the temp image
    d. Generate textures & PBR material settings for the object
2. Now, we can reconstruct the background (ie. non-objects) as a point-cloud of data using depth estimation and pixel position.  May need Concave Hull/Gift Wrapping algorithm to find edges
3. New algorithm - lighting
4. Finally, upload the results into Blender/BabylonJS/UnrealEngine to render