class DirichletValues:
    def __init__(self, homotopyParameter):
        self.homotopyParameter = homotopyParameter

    def deformation(self, x):
        out = [x[0], x[1], 0]
        if x[1] >  0.128-1e-4 :
            out[0] += 0.003 * self.homotopyParameter
        return out

    def orientation(self, x):
        rotation = [[1,0,0], [0, 1, 0], [0, 0, 1]]
        return rotation
