# Converting from Camera Space to Real Space

When we get the position of an object, what we're getting is an X and Y value in pixels, along with a depth estimate.  But that's not really an easy conversion into real-space.

Right now, we've just got a straight conversion factor of 0.001, but that's just not correct.