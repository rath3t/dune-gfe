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
parameterSet.lower = "0 0"
parameterSet.upper = "0.38 0.128"
parameterSet.elements = "15 5"

# Number of grid levels
parameterSet.numLevels = 1

#############################################
#  Solver parameters
#############################################

# Number of homotopy steps for the Dirichlet boundary conditions
parameterSet.numHomotopySteps = 1

# Tolerance of the trust region solver
parameterSet.tolerance = 1e-8

# Max number of steps of the trust region solver
parameterSet.maxSolverSteps = 500

parameterSet.solverScaling = "1 1 1 0.01 0.01 0.01"

# Initial trust-region radius
parameterSet.initialTrustRegionRadius = 0.001

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


# Parameters for the shearing/wrinkling example from Wong/Pellegrino 2006
# We use 'meters' as the length unit
parameterSet.materialParameters = ParameterSet()

# shell thickness
parameterSet.materialParameters.thickness = 2.5e-5

# Lame parameters
# corresponds to E = 3.5GPa, nu=0.31
parameterSet.materialParameters.mu = 5.6452e+09
parameterSet.materialParameters["lambda"] = 2.1796e+09

# Cosserat couple modulus
parameterSet.materialParameters.mu_c = 0

# Length scale parameter
parameterSet.materialParameters.L_c = 2.5e-8

# Curvature exponent
parameterSet.materialParameters.q = 2

# Shear correction factor
parameterSet.materialParameters.kappa = 1


#############################################
#  Boundary values
#############################################

###  Python predicate specifying all Dirichlet grid vertices
# x is the vertex coordinate
parameterSet.dirichletVerticesPredicate = "[x[1] < 0.0001 or x[1] > 0.128 - 0.0001, x[1] < 0.0001 or x[1] > 0.128 - 0.0001, x[1] < 0.0001 or x[1] > 0.128 - 0.0001]"
parameterSet.dirichletRotationVerticesPredicate = "x[1] < 0.0001 or x[1] > 0.128 - 0.0001"

### The actual Dirichlet values
# With 'homotopyParameter==0', this class gives the initial iterate
class DirichletValues:
    def __init__(self, homotopyParameter):
        self.homotopyParameter = homotopyParameter

    def deformation(self, x):
        if self.homotopyParameter<0.01:
            out = [x[0] + 0.003*x[1] / 0.128, x[1], 0.002*math.cos(1e4*x[0])]
        else:
            out = [x[0], x[1], 0]

        if x[1] >  0.128-1e-4 :
            out[0] += 0.003 * self.homotopyParameter
            out[1] += 0.0005
        return out

    def orientation(self, x):
        rotation = [[1,0,0], [0, 1, 0], [0, 0, 1]]
        return rotation

# Initial deformation
#parameterSet.startFromFile = True
parameterSet.initialIterateFilename = "cosserat_iterate_2.vtu"
