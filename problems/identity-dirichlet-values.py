import math

class DirichletValues:
    def __init__(self, homotopyParameter):
        self.homotopyParameter = homotopyParameter

    def deformation(self, x):
        return x

    def orientation(self, x):
        rotation = [[1,0,0], [0,1,0], [0,0,1]]
        return rotation
