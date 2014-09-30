class DirichletValues:
    def __init__(self, homotopyParameter):
        self.homotopyParameter = homotopyParameter

    def deformation(self, x):
        # Clamp the L-shape in its reference configuration
        return [x[0], x[1], 0]

    def orientation(self, x):
        rotation = [[1,0,0], [0, 1, 0], [0, 0, 1]]
        return rotation
