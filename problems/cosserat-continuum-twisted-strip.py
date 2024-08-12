import math

class ParameterSet(dict):
    def __init__(self, *args, **kwargs):
        super(ParameterSet, self).__init__(*args, **kwargs)
        self.__dict__ = self

parameterSet = ParameterSet()

#############################################
#  Grid parameters
#############################################

parameterSet.structuredGrid = "cube"
parameterSet.lower = "0  -0.005"
parameterSet.upper = "0.1 0.005"
parameterSet.elements = "10 1"

# Number of grid levels
parameterSet.numLevels = 1

#############################################
#  Solver parameters
#############################################

# Number of homotopy steps for the Dirichlet boundary conditions
parameterSet.numHomotopySteps = 24

# Tolerance of the trust region solver
parameterSet.tolerance = 1e-8

# Max number of steps of the trust region solver
parameterSet.maxSolverSteps = 200

parameterSet.solverScaling = "1 1 1 1 1 1"

# Initial trust-region radius
parameterSet.initialTrustRegionRadius = 0.1

# Number of multigrid iterations per trust-region step
parameterSet.numIt = 200

# Number of presmoothing steps
parameterSet.nu1 = 3

# Number of postsmoothing steps
parameterSet.nu2 = 3

# Number of coarse grid corrections
parameterSet.mu = 1

# Number of base solver iterations
parameterSet.baseIt = 100

# Tolerance of the multigrid solver
parameterSet.mgTolerance = 1e-7

# Tolerance of the base grid solver
parameterSet.baseTolerance = 1e-8

# Measure convergence
parameterSet.instrumented = 0

############################
#   Material parameters
############################


# Parameters for the twisted 0.1x0.01 strip
parameterSet.materialParameters = ParameterSet()

# shell thickness
parameterSet.materialParameters.thickness = 0.002

# Lame parameters
# corresponds to E = 3.5GPa, nu=0.31
# (Notice that lambda is a key word in python such that
# we cannot use attribute syntax to set it but have to
# resort to dictionary syntax.)
parameterSet.materialParameters.mu = 5.6452e+09
parameterSet.materialParameters['lambda'] = 2.1796e+09

# Cosserat couple modulus
parameterSet.materialParameters.mu_c = 0

# Length scale parameter
parameterSet.materialParameters.L_c = 2e-6

# Curvature exponent
parameterSet.materialParameters.q = 2

# Shear correction factor
parameterSet.materialParameters.kappa = 1

# TODO: These three parameters are not actually used,
# but the current implementation requires them nevertheless.
parameterSet.materialParameters.b1 = 1
parameterSet.materialParameters.b2 = 1
parameterSet.materialParameters.b3 = 1


#############################################
#  Boundary values
#############################################

###  Python predicate specifying all Dirichlet grid vertices
# x is the vertex coordinate
parameterSet.dirichletVerticesPredicate = "[x[0] < 0.001 or x[0] > 0.0999, x[0] < 0.001 or x[0] > 0.0999, x[0] < 0.001 or x[0] > 0.0999]"
parameterSet.dirichletRotationVerticesPredicate = "x[0] < 0.001 or x[0] > 0.0999"

### The actual Dirichlet values
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

# Initial deformation
parameterSet.initialDeformation = "[x[0], x[1], 0]"


