class DirichletValues:
    def __init__(self, homotopyParameter):
        self.homotopyParameter = homotopyParameter

    # Deformation of 3d classical materials
    def dirichletValues(self, x):
        # Clamp the L-shape in its reference configuration
        return [x[0], x[1], x[2]]

    # Deformation of Cosserat shells
    def deformation(self, x):
        # Clamp the L-shape in its reference configuration
        return [x[0], x[1], 0]

    # Orientation of Cosserat materials
    def orientation(self, x):
        rotation = [[1,0,0], [0, 1, 0], [0, 0, 1]]
        return rotation
