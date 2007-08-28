#!/bin/bash

set -e


runComputation(){

RESULTPATH=`pwd`/$1/
PARAMETERFILE=${RESULTPATH}${1}".parset"
LOGFILE=${RESULTPATH}${1}".log"

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
ddTolerance = 1e-12

# Max number of Dirichlet-Neumann steps
maxDirichletNeumannSteps = 50

# Damping for the Dirichlet-Neumann solver
damping = $3

# Tolerance of the trust-region solver for the rod problem
trTolerance = 1e-12

# Max number of steps of the trust region solver
maxTrustRegionSteps = 20

# Initial trust-region radius
initialTrustRegionRadius = 1

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
baseTolerance = 1e-8

# Measure convergence
instrumented = 1

############################
#   Problem specifications
############################

path = /home/haile/sander/data/multicoupling/simplecoupling/
gridFile = hexarod.grid
dirichletNodes = hexarod.dn
dirichletValues = hexarod.nodv
interfaceNodes  = hexarod.ifn

# Number of elements of the rod grid
numRodBaseElements = 5

# Dirichlet values
dirichletValueX = 0.5
dirichletValueY = 0.6
dirichletValueZ = 10

dirichletAxisX = 1
dirichletAxisY = 0
dirichletAxisZ = 0
dirichletAngle = 0

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
#runComputation 2levels 2


for level in 2; do
   
    for damping in 0.02; do

        runComputation 2levels $level $damping

    done

done