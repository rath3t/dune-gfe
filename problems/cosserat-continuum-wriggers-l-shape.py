import math

class ParameterSet(dict):
    def __init__(self, *args, **kwargs):
        super(ParameterSet, self).__init__(*args, **kwargs)
        self.__dict__ = self

parameterSet = ParameterSet()

#############################################
#  Grid parameters
#############################################

parameterSet.structuredGrid = "false"
parameterSet.path = "/home/sander/data/shells/wriggers_L_shape/"
parameterSet.gridFile = "wriggers-L-shape_99_mm.msh"

# Number of grid levels
parameterSet.numLevels = 2

#############################################
#  Solver parameters
#############################################

# Number of homotopy steps for the Dirichlet boundary conditions
parameterSet.numHomotopySteps = 1

# Tolerance of the trust region solver
parameterSet.tolerance = 1e-8

# Max number of steps of the trust region solver
parameterSet.maxSolverSteps = 1000

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
parameterSet.baseIt = 1

# Tolerance of the multigrid solver
parameterSet.mgTolerance = 1e-7

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
parameterSet.materialParameters.L_c = 0.6e-3

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

parameterSet.problem = "wriggers-l-shape"

###  Python predicate specifying all Dirichlet grid vertices
# x is the vertex coordinate
parameterSet.dirichletVerticesPredicate = "[x[0] < 1, x[0] < 1, x[0] < 1]"
parameterSet.dirichletRotationVerticesPredicate = "x[0] < 1"

###  Python predicate specifying all Dirichlet grid vertices
# x is the vertex coordinate
parameterSet.neumannVerticesPredicate = "x[1] < -239"

###  Neumann values, if needed
parameterSet.neumannValues =  "0.09 0 0"

# Initial deformation
#parameterSet.initialDeformation = "[x[0], x[1], 0 if (x[0] < 225 or x[1] < -15) else 0.001*(x[0]-225)*(x[1]+15)]"
##parameterSet.initialDeformation = "[x[0], x[1], 0]"

parameterSet.startFromFile = True
parameterSet.initialIterateGridFilename = "wriggers-L-shape_99_mm.msh"
parameterSet.initialIterateFilename = "initial-wriggers-l-shape.vtu"
