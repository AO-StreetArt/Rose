# Rose v1

What really *is* spatial intelligence?

Spatial intelligence, or visuo-spatial ability, has been defined “the ability to generate, retain, retrieve, and transform well-structured visual images” (Lohman 1996).

Visual-spatial intelligence is one such set of skills that includes the ability to perceive, hold, manipulate, and problem-solve from visual information. When you put together a puzzle, you use visual-spatial skills to identify which pieces have similar colors that go near each other or similar shapes that will fit together.

Dr. Richard Kraft, Ph.D. and professor of cognitive psychology at Otterbein University, says that “Visual-spatial intelligence is our ability to think about the world in three dimensions. We use visual-spatial intelligence to find our way around and to manipulate mental images of objects and the spaces these objects are in. People with strong visual-spatial intelligence have a good sense of direction, and they know how parts fit together into a whole (like assembling furniture from IKEA).”


So we're trying to answer a few questions with spatial intelligence:
* Where am I?
* Where are the things around me?
* Are they moving?  How fast and in what directions?

And that covers a base-level, "I exist and know other things exist in the world, and can react to visual stimuli" sort of intelligence.

Then, we can ask more difficult questions:
* What are the things around me?
* What might show up around me soon?
* What could exist around me? - ie. creating something that doesn't exist

## 3D Environment Storage

The outputs of the below algorithms are stored in long term storage.  This can be done locally or remotely, and deeper logic starts with the short-term storage before moving to long-term storage.

This storage could take several forms, but should account for elements/features detected, and their relative size, position, and movement in 3D space.

The retention in short-term storage, movement to long-term storage, and finally long-term archival is a complex process in-and-of-itself.  We want to keep things longer that are frequently referenced, and cache long-term items into the short-term storage upon access.  Archival only happens once a feature has been categorized and analyzed, and is ready to be put away for very slow recall.

## Basic algorithm

Rose takes a short series of images and tries to answer the following questions:
* What elements are in the images
* Have we seen those elements before?
* What elements are the same and different between the images
* How are elements in the image moving between frames

First, Rose looks at each image and determines:
* What is in the image?
* Where are the elements in the image relative to the camera?

Then, the sets of elements in each frame are compared, and we can determine what elements are shared vs unique.
For shared elements, the relative positions of each are then put through a basic 3D distance function to return the movement vector.  If given a framerate, this can then be multiplied to give a velocity.
