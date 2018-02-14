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

# Takes a 3 x 3 object, returns a 4 x 3 x 3 object
def derivativeOfMatrixToQuaternion(m):

    result = [[[0,0,0],
               [0,0,0],
               [0,0,0]],
              [[0,0,0],
               [0,0,0],
               [0,0,0]],
              [[0,0,0],
               [0,0,0],
               [0,0,0]],
              [[0,0,0],
               [0,0,0],
               [0,0,0]]]

    p = [(1 + m[0][0] - m[1][1] - m[2][2]) / 4,
         (1 - m[0][0] + m[1][1] - m[2][2]) / 4,
         (1 - m[0][0] - m[1][1] + m[2][2]) / 4,
         (1 + m[0][0] + m[1][1] + m[2][2]) / 4]

    # avoid rounding problems
    if (p[0] >= p[1] and p[0] >= p[2] and p[0] >= p[3]):

        result[0] = [[1,0,0],[0,-1,0],[0,0,-1]]
        for i in range(0,3):
            for j in range(0,3):
                result[0][i][j] *= 1.0/(8.0*sqrt(p[0]))

        denom = 32 * pow(p[0],1.5)
        offDiag = 1.0/(4*sqrt(p[0]))

        result[1] = [ [-(m[0][1]+m[1][0]) / denom, offDiag,              0],
                      [offDiag,                    (m[0][1]+m[1][0]) / denom, 0],
                      [0,                          0,                         (m[0][1]+m[1][0]) / denom]]

        result[2] = [ [-(m[0][2]+m[2][0]) / denom, 0,                        offDiag],
                        [0,                         (m[0][2]+m[2][0]) / denom, 0],
                        [offDiag,                    0,                       (m[0][2]+m[2][0]) / denom]]

        result[3] = [ [-(m[2][1]-m[1][2]) / denom, 0,                         0],
                      [0,                         (m[2][1]-m[1][2]) / denom, -offDiag],
                      [0,                          offDiag,                   (m[2][1]-m[1][2]) / denom]]

    elif (p[1] >= p[0] and p[1] >= p[2] and p[1] >= p[3]):

        result[1] = [[-1,0,0],[0,1,0],[0,0,-1]]
        for i in range(0,3):
            for j in range(0,3):
                result[1][i][j] *= 1.0/(8.0*sqrt(p[1]))

        denom = 32 * pow(p[1],1.5)
        offDiag = 1.0/(4*sqrt(p[1]))

        result[0] = [ [(m[0][1]+m[1][0]) / denom, offDiag,                    0],
                      [offDiag,                  -(m[0][1]+m[1][0]) / denom, 0],
                      [0,                         0,                         (m[0][1]+m[1][0]) / denom]]

        result[2] = [ [(m[1][2]+m[2][1]) / denom, 0           ,              0],
                      [0,                        -(m[1][2]+m[2][1]) / denom, offDiag],
                      [0,                         offDiag,                   (m[1][2]+m[2][1]) / denom]]

        result[3] = [ [(m[0][2]-m[2][0]) / denom, 0,                         offDiag],
                      [0,                        -(m[0][2]-m[2][0]) / denom, 0],
                      [-offDiag,                  0,                         (m[0][2]-m[2][0]) / denom]]

    elif (p[2] >= p[0] and p[2] >= p[1] and p[2] >= p[3]):

        result[2] = [[-1,0,0],[0,-1,0],[0,0,1]]
        for i in range(0,3):
            for j in range(0,3):
                result[2][i][j] *= 1.0/(8.0*sqrt(p[2]))

        denom = 32 * pow(p[2],1.5)
        offDiag = 1.0/(4*sqrt(p[2]))

        result[0] = [ [(m[0][2]+m[2][0]) / denom, 0,                         offDiag],
                      [0,                        (m[0][2]+m[2][0]) / denom, 0],
                      [offDiag,                   0,                         -(m[0][2]+m[2][0]) / denom]]

        result[1] = [ [(m[1][2]+m[2][1]) / denom, 0           ,              0],
                      [0,                        (m[1][2]+m[2][1]) / denom,  offDiag],
                      [0,                         offDiag,                 -(m[1][2]+m[2][1]) / denom]]

        result[3] = [ [(m[1][0]-m[0][1]) / denom, -offDiag,                  0],
                      [offDiag,                   (m[1][0]-m[0][1]) / denom, 0],
                      [0,                          0,                        -(m[1][0]-m[0][1]) / denom]]

    else:

        result[3] = [[1,0,0],[0,1,0],[0,0,1]]
        for i in range(0,3):
            for j in range(0,3):
                result[3][i][j] *= 1.0/(8.0*sqrt(p[3]))

        denom = 32 * pow(p[3],1.5)
        offDiag = 1.0/(4*sqrt(p[3]))

        result[0] = [ [-(m[2][1]-m[1][2]) / denom, 0,                         0],
                      [0,                         -(m[2][1]-m[1][2]) / denom, -offDiag],
                      [0,                         offDiag,                   -(m[2][1]-m[1][2]) / denom]]

        result[1] = [ [-(m[0][2]-m[2][0]) / denom, 0,                         offDiag],
                      [0,                         -(m[0][2]-m[2][0]) / denom, 0],
                      [-offDiag,                  0,                         -(m[0][2]-m[2][0]) / denom]]

        result[2] = [ [-(m[1][0]-m[0][1]) / denom, -offDiag,                  0],
                      [offDiag,                   -(m[1][0]-m[0][1]) / denom, 0],
                      [0,                         0,                         -(m[1][0]-m[0][1]) / denom]]

    return result



def mult(m1,m2):
    m = [[0,0,0],
         [0,0,0],
         [0,0,0]]

    # Matrix-matrix multiplication by hand -- for some reason I cannot get numpy to work
    for i in range(0,3):
        for j in range(0,3):
            for k in range(0,3):
                m[i][j] = m[i][j] + m1[i][k] * m2[k][j]

    return m

# Scale the domain by this factor
scale = 0.2 * math.pi

def f(x):

    alpha = scale*x[0]
    beta  = scale*x[1]
    # Dependency on third coordinate only if it exists
    if len(x)==3:
        gamma = scale*x[2]
    else:
        gamma = 0

    rotationX = [[1, 0, 0],
                 [0, cos(alpha), -sin(alpha)],
                 [0, sin(alpha),  cos(alpha)]]

    rotationY = [[cos(beta), 0, -sin(beta)],
                 [0,          1, 0],
                 [sin(beta), 0,  cos(beta)]]

    rotationZ = [[cos(gamma), -sin(gamma), 0],
                 [sin(gamma),  cos(gamma), 0],
                 [0, 0, 1]]

    m = mult(rotationX,rotationY)
    m = mult(m,rotationZ)

    return matrixToQuaternion(m)

def df(x):

    alpha = scale*x[0]
    beta  = scale*x[1]
    # Dependency on third coordinate only if it exists
    if len(x)==3:
        gamma = scale*x[2]
    else:
        gamma = 0

    rotationX = [[1, 0, 0],
                 [0, cos(alpha), -sin(alpha)],
                 [0, sin(alpha),  cos(alpha)]]

    derRotX   = [[0, 0, 0],
                 [0, -scale*sin(alpha), -scale*cos(alpha)],
                 [0,  scale*cos(alpha), -scale*sin(alpha)]]

    rotationY = [[cos(beta), 0, -sin(beta)],
                 [0,          1, 0],
                 [sin(beta), 0,  cos(beta)]]

    derRotY   = [[-scale*sin(beta), 0, -scale*cos(beta)],
                 [0,          0, 0],
                 [ scale*cos(beta), 0, -scale*sin(beta)]]

    rotationZ = [[cos(gamma), -sin(gamma), 0],
                 [sin(gamma),  cos(gamma), 0],
                 [0, 0, 1]]

    # Dependency on third coordinate only if it exists
    if len(x)==3:
        derRotZ   = [[-scale*sin(gamma), -scale*cos(gamma), 0],
                     [ scale*cos(gamma), -scale*sin(gamma), 0],
                     [0, 0, 0]]
    else:
        derRotZ   = [[0, 0, 0],
                     [0, 0, 0],
                     [0, 0, 0]]

    # The derivative in matrix space
    derX0 = mult(derRotX,rotationY)
    derX0 = mult(derX0,rotationZ)

    derX1 = mult(rotationX,derRotY)
    derX1 = mult(derX1,rotationZ)

    derX2 = mult(rotationX,rotationY)
    derX2 = mult(derX2,derRotZ)

    # The function value as matrix
    # CAREFUL! We are reimplementing method 'f' here -- the two implementations need to match!
    value = mult(rotationX,rotationY)
    value = mult(value,rotationZ)

    derOfMatrixToQuaternion = derivativeOfMatrixToQuaternion(value)

    if (len(x)==3):
        result = [[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
    else:
        result = [[0,0],[0,0],[0,0],[0,0]]


    for i in range(0,4):
        for j in range(0,3):
            for k in range(0,3):
                result[i][0] += derOfMatrixToQuaternion[i][j][k] * derX0[j][k]
                result[i][1] += derOfMatrixToQuaternion[i][j][k] * derX1[j][k]
                if (len(x)==3):
                    result[i][2] += derOfMatrixToQuaternion[i][j][k] * derX2[j][k]

    return result

fdf = (f, df)
