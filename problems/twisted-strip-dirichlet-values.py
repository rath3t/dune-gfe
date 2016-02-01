import math

class DirichletValues:
    def __init__(self, homotopyParameter):
        self.homotopyParameter = homotopyParameter
        self.upper = [0.1, 0.01]
        self.totalAngle = 6*math.pi

    def deformation(self, x):
        angle = self.totalAngle * x[0]/self.upper[0]
        angle *= self.homotopyParameter

        # Rotation matrix (around y-axis)
        rotation = [[1,0,0], [0, math.cos(angle), -math.sin(angle)], [0, math.sin(angle), math.cos(angle)]]

        # Matrix-vector product, vector is [x[0], x[1], 0]
        out = [rotation[0][0]*x[0]+rotation[0][1]*x[1], rotation[1][0]*x[0]+rotation[1][1]*x[1], rotation[2][0]*x[0]+rotation[2][1]*x[1]]

        return out


    def orientation(self, x):
        angle = self.totalAngle * x[0]/self.upper[0]
        angle *= self.homotopyParameter

        rotation = [[1,0,0], [0, math.cos(angle), -math.sin(angle)], [0, math.sin(angle), math.cos(angle)]]
        return rotation

