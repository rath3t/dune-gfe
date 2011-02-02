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

# Type of the domain decomposition algorithm
# Currently supported:
#  - FixedPointIteration
#  - RichardsonIteration
ddType = RichardsonIteration
#ddType = FixedPointIteration

# What is the preconditioner, if we are doing a Richardson iteration?
# Possible values:
#  - DirichletNeumann
#  - NeumannDirichlet
#  - NeumannNeumann
#  - RobinRobin
preconditioner = $4

#
NeumannNeumannDamping = 0.5 0.5

# Tolerance of the Dirichlet-Neumann solver
ddTolerance = 1e-9

# Max number of Dirichlet-Neumann steps
maxDirichletNeumannSteps = 100

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

#path = /home/sander/data/multicoupling/simplecoupling/
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

#rodRestEndPoint0 = 0.125 0.125 1
#rodRestEndPoint1 = 0.125 0.125 2

## Dirichlet values
#dirichletValue = 0.125 0.125 2.5

#dirichletAxis = 0 0 1
#dirichletAngle = 0


########################################################
#path = /home/sander/data/multicoupling/simplecoupling/
#gridFile = cube_5x5x5.grid
#dirichletNodes = cube_5x5x5.dn
#dirichletValues = cube_5x5x5.nodv
#interfaceNodes  = cube_5x5x5.ifn

#numRodBaseElements = 5

# Cross-section area
#rodA = 1

# Geometric moments (here: square of edge length one)
#rodJ1 = 0.0833333
#rodJ2 = 0.0833333

# Material parameters
#rodE  = 2.5e5
#rodNu = 0.3
#E     = 1e6
#nu    = 0.3

## Dirichlet values
#dirichletValue = 0.5 1.5 10

#dirichletAxis = 0 0 1
#dirichletAngle = 90

########################################################
path = /home/sander/data/multicoupling/simplecoupling/
gridFile = cube_4x4x4.grid
dirichletNodes = cube_4x4x4.dn
dirichletValues = cube_4x4x4.nodv
interfaceNodes  = cube_4x4x4.ifn

numRodBaseElements = 4

# Cross-section area
rodA = 0.0625

# Geometric moments (here: square of edge length 0.25)
#rodJ1 = 0.0013021
#rodJ2 = 0.0013021
rodJ1 = 0.00032552
rodJ1 = 0.00032552

# Material parameters
rodE  = 1e6
rodNu = 0.3
E     = 1e6
nu    = 0.3

rodRestEndPoint0 = 0.625 0.625 1
rodRestEndPoint1 = 0.625 0.625 2
#rodRestEndPoint1 = 0.625 -0.082 1.707

# Dirichlet values
dirichletValue1 = 0.625 0.875 2.0

dirichletAxis1 = 0 0 1
dirichletAngle1 = 90

# Where to write the results
resultPath = $RESULTPATH

EOF

mkdir -p ${RESULTPATH}"/tmp_"$3

# run simulation
../dirneucoupling ${PARAMETERFILE} | tee ${LOGFILE}

rm -rf ${RESULTPATH}"/tmp_"$3
}

# Parameters:
# 1: result directory
# 2: number of levels
# 3: damping factor
# 4: preconditioner

# run problems

MAXPROCS=4

for level in 1; do
#for level in 1 2 3 4; do

    
#     if test -e ${LEVELDIR}/convrates; then
#         rm ${LEVELDIR}/convrates
#     fi
    preconditioner=NeumannNeumann
    LEVELDIR=${preconditioner}"_"${level}"levels"
#  for damping in 0.1; do
#    for damping in 0.1 0.6 0.9; do
    for damping in 0.001 0.005 0.01 0.05 0.1 0.2 2.25 2.5 2.75 0.3 0.35; do

        echo "Computing on "${level}" levels with damping factor "${damping}
        runComputation $LEVELDIR $level $damping $preconditioner &

        # Append convergence rate of this run to overall list for this level
        #cat convrate >> ${LEVELDIR}/convrates

        # Never have more than MAXPROCS processes
        NPROC=$(($NPROC+1))  
        if [ "$NPROC" -ge "$MAXPROCS" ]; then  
          wait  
          NPROC=0  
        fi  

    done



    preconditioner=DirichletNeumann
    LEVELDIR=${preconditioner}"_"${level}"levels"
    for damping in 0.001 0.005 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do

        echo "Computing on "${level}" levels with damping factor "${damping}
        runComputation $LEVELDIR $level $damping $preconditioner &

        # Append convergence rate of this run to overall list for this level
        #cat convrate >> ${LEVELDIR}/convrates

        # Never have more than MAXPROCS processes
        NPROC=$(($NPROC+1))  
        if [ "$NPROC" -ge "$MAXPROCS" ]; then  
          wait  
          NPROC=0  
        fi  

    done

    preconditioner=NeumannDirichlet
    LEVELDIR=${preconditioner}"_"${level}"levels"
    #for damping in 0.115; do
    for damping in 0.001 0.005 0.01 0.05 0.1 0.2 2.25 2.5 2.75 0.3 0.35; do

        echo "Computing on "${level}" levels with damping factor "${damping}
        runComputation $LEVELDIR $level $damping $preconditioner &

        # Append convergence rate of this run to overall list for this level
        #cat convrate >> ${LEVELDIR}/convrates

        # Never have more than MAXPROCS processes
        NPROC=$(($NPROC+1))  
        if [ "$NPROC" -ge "$MAXPROCS" ]; then  
          wait  
          NPROC=0  
        fi  

    done

done