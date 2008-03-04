#!/bin/bash

set -e


runComputation(){

RESULTPATH=`pwd`/$1/
PARAMETERFILE=${RESULTPATH}${1}"_"${3}".parset"
LOGFILE=${RESULTPATH}${1}"_"${3}".log"

#echo $RESULTPATH

# Set up directory where to store the results
if ! test -d "$RESULTPATH"; then
    mkdir $RESULTPATH
fi
#rm $RESULTPATH/*

cat > "${PARAMETERFILE}" <<EOF
# Number of grid levels
numLevels = $2

# Tolerance of the Dirichlet-Neumann solver
ddTolerance = 1e-9

# Max number of Dirichlet-Neumann steps
maxDirichletNeumannSteps = 30

# Tolerance of the trust-region solver for the rod problem
trTolerance = 1e-12

# Max number of steps of the trust region solver
maxTrustRegionSteps = 20

# Verbosity of the trust-region solver
trVerbosity = 0

# Number of multigrid iterations per trust-region step
numIt = 30

# Number of presmoothing steps
nu1 = 3

# Number of postsmoothing steps
nu2 = 3

# Number of coarse grid corrections
mu = 1

# Number of base solver iterations
baseIt = 100

# Tolerance of the multigrid solver
mgTolerance = 1e-13

# Tolerance of the base grid solver
baseTolerance = 1e-13

# Initial trust-region radius
initialTrustRegionRadius = 1

# Damping
damping = $3

# Measure convergence
instrumented = 0

############################
#   Problem specifications
############################

#path = /home/haile/sander/data/multicoupling/simplecoupling/
#gridFile = hexarod.grid
#dirichletNodes = hexarod.dn
#dirichletValues = hexarod.nodv
#interfaceNodes  = hexarod.ifn

#numRodBaseElements = 4

## Cross-section area
#rodA = 0.0625

## Geometric moments (here: square of edge length 0.25)
#rodJ1 = 0.0013021
#rodJ2 = 0.0013021

## Material parameters
#rodE  = 1e6
#rodNu = 0.3

#E     = 1e6
#nu    = 0.3

#rodRestEndPoint0X = 0.125
#rodRestEndPoint0Y = 0.125
#rodRestEndPoint0Z = 1
#rodRestEndPoint1X = 0.125
#rodRestEndPoint1Y = 0.125
#rodRestEndPoint1Z = 2

## Dirichlet values
#dirichletValueX = 0.125
#dirichletValueY = 0.125
#dirichletValueZ = 2.5

#dirichletAxisX = 0
#dirichletAxisY = 0
#dirichletAxisZ = 1
#dirichletAngle = 0


########################################################
#path = /home/haile/sander/data/multicoupling/simplecoupling/
#gridFile = cube_5x5x5.grid
#dirichletNodes = cube_5x5x5.dn
#dirichletValues = cube_5x5x5.nodv
#interfaceNodes  = cube_5x5x5.ifn

#numRodBaseElements = 5

# Cross-section area
#rodA = 1

# Geometric moments (here: square of edge length one)
#rod J1 = 0.0833333
#rod J2 = 0.0833333

# Material parameters
#rodE  = 2.5e5
#rodNu = 0.3

## Dirichlet values
#dirichletValueX = 0.5
#dirichletValueY = 1.5
#dirichletValueZ = 10

#dirichletAxisX = 0
#dirichletAxisY = 0
#dirichletAxisZ = 1
#dirichletAngle = 90

########################################################
path = /home/haile/sander/data/multicoupling/simplecoupling/
gridFile = cube_4x4x4.grid
dirichletNodes = cube_4x4x4.dn
dirichletValues = cube_4x4x4.nodv
interfaceNodes  = cube_4x4x4.ifn

numRodBaseElements = 4

# Cross-section area
rodA = 0.0625

# Geometric moments (here: square of edge length 0.25)
rodJ1 = 0.0013021
rodJ2 = 0.0013021

# Material parameters
rodE  = 1e6
rodNu = 0.3
E     = 1e6
nu    = 0.3

rodRestEndPoint0X = 0.625
rodRestEndPoint0Y = 0.625
rodRestEndPoint0Z = 1
#rodRestEndPoint1X = 0.625
#rodRestEndPoint1Y = 0.625
#rodRestEndPoint1Z = 2
rodRestEndPoint1X = 0.625
rodRestEndPoint1Y = -0.082
rodRestEndPoint1Z = 1.707

# Dirichlet values
dirichletValueX = 0.625
dirichletValueY = 0.875
dirichletValueZ = 2

dirichletAxisX = 0
dirichletAxisY = 0
dirichletAxisZ = 1
dirichletAngle = 90

# Where to write the results
resultPath = $RESULTPATH

EOF

# run simulation
../dirneucoupling ${PARAMETERFILE} | tee ${LOGFILE}
}

# Parameters:
# 1: result directory
# 2: number of levels
# 3: damping factor

# run problems


for level in 4; do
#for level in 1 2 3 4; do

    LEVELDIR=${level}"levels_rot"
    
    if test -e ${LEVELDIR}/convrates; then
        rm ${LEVELDIR}/convrates
    fi

    for damping in 0.6; do
#    for damping in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3; do

        echo "Computing on "${level}" levels with damping factor "${damping}
        runComputation $LEVELDIR $level $damping

        # Append convergence rate of this run to overall list for this level
        #cat convrate >> ${LEVELDIR}/convrates

    done

done