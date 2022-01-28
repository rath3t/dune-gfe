import math

class DirichletValues:
    def __init__(self, homotopyParameter):
        self.homotopyParameter = homotopyParameter

    def deformation(self, x):
        # Dirichlet b.c. simply clamp the shell in the reference configuration
        out = [x[0], x[1], 0]

        return out

    def director(self, x):
        dir = [0, 0, 1]
        return dir

