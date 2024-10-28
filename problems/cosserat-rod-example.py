import math

class ParameterSet(dict):
    def __init__(self, *args, **kwargs):
        super(ParameterSet, self).__init__(*args, **kwargs)
        self.__dict__ = self

parameterSet = ParameterSet()

# Interpolation method
parameterSet.interpolationMethod = "geodesic"

#############################################
#  Grid parameters
#############################################

# Number of grid levels
parameterSet.numLevels = 4

# Number of elements of the rod grid
parameterSet.numRodBaseElements = 5

#############################################
#  Solver parameters
#############################################

# Initial load increment
parameterSet.loadIncrement = 0.01

# Tolerance of the trust region solver
parameterSet.tolerance = 1e-6

# Max number of steps of the trust region solver
parameterSet.maxTrustRegionSteps = 100

parameterSet.solverScaling = "1 1 1 0.01 0.01 0.01"

# Initial trust-region radius
parameterSet.initialTrustRegionRadius = 1

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
parameterSet.mgTolerance = 1e-10

# Tolerance of the base grid solver
parameterSet.baseTolerance = 1e-8

parameterSet.instrumented = "no"


############################
#   Material parameters
############################

parameterSet.A = 1e-4
parameterSet.J1 = 1e-4
parameterSet.J2 = 1e-4
parameterSet.E = 2.5e5
parameterSet.nu = 0.3

#############################################
#  Boundary values
#############################################

class ReferenceConfiguration:
    def deformation(self, x):
        return [x[0],0,0]

    def orientation(self, x):
        return [[1,0,0], [0,1,0], [0,0,1]]

###  Python predicate specifying all Dirichlet grid vertices
# x is the vertex coordinate
parameterSet.dirichletVerticesPredicate = "[x[0] < 0.001 or x[0] > 0.999, \
                                            x[0] < 0.001 or x[0] > 0.999, \
                                            x[0] < 0.001 or x[0] > 0.999]"
parameterSet.dirichletRotationVerticesPredicate = "x[0] < 0.001 or x[0] > 0.999"

# Dirichlet values and initial iterate
class DirichletValues:
    def deformation(self, x):
        return [x[0]*0.5, 0, 0.1*math.sin(x[0]*math.pi)]

    def orientation(self, x):
        return [[1,0,0], [0,1,0], [0,0,1]]
