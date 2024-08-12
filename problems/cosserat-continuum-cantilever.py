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

# bounding box
parameterSet.lower = "0 0"
parameterSet.upper = "100 10"

parameterSet.elements = "10 1"

# Number of grid levels
parameterSet.numLevels = 1

#############################################
#  Solver parameters
#############################################

# Number of homotopy steps for the Dirichlet boundary conditions
parameterSet.numHomotopySteps = 1

# Solver type: "trustRegion" or "proximalNewton"
parameterSet.solvertype = "trustRegion"

# Tolerance of the trust region solver
parameterSet.tolerance = 1e-3

# Max number of steps of the trust region solver
parameterSet.maxSolverSteps = 200

parameterSet.solverScaling = "1 1 1 0.01 0.01 0.01"

# Initial trust-region radius
parameterSet.initialTrustRegionRadius = 3.125

# Number of multigrid iterations per trust-region step
parameterSet.numIt = 400

# Number of presmoothing steps
parameterSet.nu1 = 3

# Number of postsmoothing steps
parameterSet.nu2 = 3

# Number of coarse grid corrections
parameterSet.mu = 1

# Number of base solver iterations
parameterSet.baseIt = 1

# Tolerance of the multigrid solver
parameterSet.mgTolerance = 1e-5

# Tolerance of the base grid solver
parameterSet.baseTolerance = 1e-8

# Measure convergence
parameterSet.instrumented = 0

############################
#   Material parameters
############################

parameterSet.materialParameters = ParameterSet()

# shell thickness
parameterSet.materialParameters.thickness = 0.6

# Lame parameters
# corresponds to E = 71240 N/mm^2, nu=0.31
# However, we use units N/m^2
parameterSet.materialParameters.mu = 2.7191e+4
parameterSet.materialParameters["lambda"] = 4.4364e+4

# Cosserat couple modulus
parameterSet.materialParameters.mu_c = 0

# Length scale parameter
parameterSet.materialParameters.L_c = 0.6

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
parameterSet.dirichletVerticesPredicate = "[x[0] < 0.01, x[0] < 0.01, x[0] < 0.01]"
parameterSet.dirichletRotationVerticesPredicate = "x[0] < 0.01"

### The actual Dirichlet values
class DirichletValues:
    def __init__(self, homotopyParameter):
        self.homotopyParameter = homotopyParameter

    def deformation(self, x):
        # Dirichlet b.c. simply clamp the shell in the reference configuration
        out = [x[0], x[1], 0]

        return out


    def orientation(self, x):
        rotation = [[1,0,0], [0, 1, 0], [0, 0, 1]]
        return rotation

###  Python predicate specifying all Neumann grid vertices
# x is the vertex coordinate
parameterSet.neumannVerticesPredicate = "x[0] > 99.99"

###  Neumann values
parameterSet.neumannValues = "0 0 3"

# Initial deformation
parameterSet.initialDeformation = "[x[0], x[1], 0]"

#parameterSet.startFromFile = yes
#parameterSet.initialIterateFilename = initial_iterate.vtu
