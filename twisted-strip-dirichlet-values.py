import math

class DirichletValues:
    def __init__(self, homotopyParameter):
        self.homotopyParameter = homotopyParameter
        self.upper = [0.1, 0.01]
        self.totalAngle = 6*math.pi

    def deformation(self, x):
        angle = self.totalAngle * x[0]/self.upper[0]
        angle *= self.homotopyParameter

        # Center of rotation
        center = [0, 0, 0]
        center[1] = self.upper[1]/2.0

        rotation = [[1,0,0], [0, math.cos(angle), -math.sin(angle)], [0, math.sin(angle), math.cos(angle)]]

        inEmbedded = [x[0]-center[0], x[1]-center[1], 0-center[2]]

        # Matrix-vector product
        out = [rotation[0][0]*inEmbedded[0]+rotation[0][1]*inEmbedded[1], rotation[1][0]*inEmbedded[0]+rotation[1][1]*inEmbedded[1], rotation[2][0]*inEmbedded[0]+rotation[2][1]*inEmbedded[1]]

        out = [out[0]+center[0], out[1]+center[1], out[2]+center[2]]
        return out


    def orientation(self, x):
        angle = self.totalAngle * x[0]/self.upper[0]
        angle *= self.homotopyParameter

        rotation = [[1,0,0], [0, math.cos(angle), -math.sin(angle)], [0, math.sin(angle), math.cos(angle)]]
        return rotation

