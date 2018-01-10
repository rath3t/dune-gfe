# A synthetic map from the unit cube into SO(3), for approximation error testing
from math import sin
from math import cos
from math import sqrt
import math

# The following equations for the construction of a unit quaternion from a rotation
# matrix comes from 'E. Salamin, Application of Quaternions to Computation with
# Rotations, Technical Report, Stanford, 1974'
def matrixToQuaternion(m):
    p = [0,0,0,0]

    p[0] = (1 + m[0][0] - m[1][1] - m[2][2]) / 4
    p[1] = (1 - m[0][0] + m[1][1] - m[2][2]) / 4
    p[2] = (1 - m[0][0] - m[1][1] + m[2][2]) / 4
    p[3] = (1 + m[0][0] + m[1][1] + m[2][2]) / 4

    # avoid rounding problems
    if (p[0] >= p[1] and p[0] >= p[2] and p[0] >= p[3]):

        p[0] = sqrt(p[0])
        p[1] = (m[0][1] + m[1][0]) / 4 / p[0]
        p[2] = (m[0][2] + m[2][0]) / 4 / p[0]
        p[3] = (m[2][1] - m[1][2]) / 4 / p[0]

    elif (p[1] >= p[0] and p[1] >= p[2] and p[1] >= p[3]):

        p[1] = sqrt(p[1])
        p[0] = (m[0][1] + m[1][0]) / 4 / p[1]
        p[2] = (m[1][2] + m[2][1]) / 4 / p[1]
        p[3] = (m[0][2] - m[2][0]) / 4 / p[1]

    elif (p[2] >= p[0] and p[2] >= p[1] and p[2] >= p[3]):

        p[2] = sqrt(p[2])
        p[0] = (m[0][2] + m[2][0]) / 4 / p[2]
        p[1] = (m[1][2] + m[2][1]) / 4 / p[2]
        p[3] = (m[1][0] - m[0][1]) / 4 / p[2]

    else:

        p[3] = sqrt(p[3])
        p[0] = (m[2][1] - m[1][2]) / 4 / p[3]
        p[1] = (m[0][2] - m[2][0]) / 4 / p[3]
        p[2] = (m[1][0] - m[0][1]) / 4 / p[3]

    return p



def f(x):

    alpha = 2*math.pi*x[0]
    beta  = 2*math.pi*x[1]

    rotationX = [[1, 0, 0],
                 [0, cos(alpha), -sin(alpha)],
                 [0, sin(alpha),  cos(alpha)]]

    rotationZ = [[cos(beta), -sin(beta), 0],
                 [sin(beta),  cos(beta), 0],
                 [0, 0, 1]]

    m = [[0,0,0],
         [0,0,0],
         [0,0,0]]

    # Matrix-matrix multiplication by hand -- for some reason I cannot get numpy to work
    for i in range(0,3):
        for j in range(0,3):
            for k in range(0,3):
                m[i][j] = m[i][j] + rotationX[i][k] * rotationZ[k][j]

    return matrixToQuaternion(m)

# Should never be called
def df(x):
    return [[0,0],[0,0],[0,0],[0,0]]

fdf = (f, df)
