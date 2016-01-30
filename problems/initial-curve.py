import math

# Wiggly initial curve on the sphere in R^3 for a curve-shortening flow simulation
#
# Note: This method does not have to return unit vectors.  Any non-zero vectors
# will do; they are normalized after reading anyway.
def f(x):
    return [math.sin(0.5*math.pi*x[0]), math.cos(0.5*math.pi*x[0]), math.sin(10*math.pi*x[0])]

