import math

# Wiggly initial curve on the sphere in R^3 for a curve-shortening flow simulation
def f(x):
    return [math.sin(0.5*math.pi*x[0]), math.cos(0.5*math.pi*x[0]), 0]

