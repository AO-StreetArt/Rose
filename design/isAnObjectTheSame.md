# Is one image of the same subject as another?

So we've got two images - one from some amount of time ago (ie. a few milliseconds, a few seconds, etc), and the other from now.  How do we determine if they are of the same thing?

Right now, we're using an Image Comparison algorithm, but this is really imperfect.  It's also performance intensive, as it needs to be run on each frame.  

## Facial Recognition

When we have a frontal view of a person, we can drop into a sub-routine of facial recognition instead of Image Comparison.