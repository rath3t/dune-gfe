import math

class ParameterSet(dict):
    def __init__(self, *args, **kwargs):
        super(ParameterSet, self).__init__(*args, **kwargs)
        self.__dict__ = self

parameterSet = ParameterSet()

#############################################
#  Paths
#############################################
parameterSet.resultPath = '/home/klaus/Desktop/Dune-Hermite/dune-gfe/outputs_L-clamped-Plate'
parameterSet.instrumentedPath = '/home/klaus/Desktop/Dune-Hermite/dune-gfe/outputs_L-clamped-Plate/instrumented'
parameterSet.executablePath = '/home/klaus/Desktop/Dune-Hermite/dune-gfe/build-cmake/src'
parameterSet.baseName= 'L-clamped-Plate'

#############################################
#  Grid parameters
#############################################
nX=1
nY=1

parameterSet.structuredGrid = 'simplex'
parameterSet.lower = '0 0'
parameterSet.upper = '4 4'
parameterSet.elements = str(nX)+' '+  str(nY)

parameterSet.macroGridLevel = 3
#############################################
#  Solver parameters
#############################################
# Choose solver: "RNHM" (Default:Riemannian Newton with Hessian modification), "RiemannianTR" (Riemannain Trust-region method)
parameterSet.Solver = "RNHM"
# Tolerance of the multigrid solver
parameterSet.tolerance = 1e-12
# Maximum number of multigrid iterations
parameterSet.maxProximalNewtonSteps = 100
# Initial regularization
parameterSet.initialRegularization = 1
# Measure convergence
parameterSet.instrumented = 0

############################
#   Problem specifications
############################
# Dimension of the domain (only used for numerical-test python files)
parameterSet.dim = 2

# Dimension of the target space
parameterSet.targetDim = 3

parameterSet.targetSpace = 'BendingIsometry'
#############################################
#  Options
#############################################
# Write discrete solution as .vtk-File
parameterSet.writeVTK = 1

# Write Dof-Vector to .txt-file
parameterSet.writeDOFvector = 0

#############################################
#  Dirichlet boundary indicator
#############################################
def dirichlet_indicator(x) :
    if( (x[0] <= 0.001) or (x[1]<=0.001)):
        return True
    else:
        return False


#############################################
#  Initial iterate function
#############################################
def f(x):
    return [x[0], x[1], 0]


def df(x):
    return ((1,0),
            (0,1),
            (0,0))


fdf = (f, df)

#############################################
#  Force
############################################
def force(x):
    return [0, 0, 0.025]




