#!/bin/bash

set -e

runComputation(){

numLevels=$1
deformationOrder=$2
rotationOrder=$3

#RESULTPATH=`pwd`/richards_surfacewater_results_${leakage}_${richardsonDamping}/
#LOGFILE=${RESULTPATH}/"richards_${mu_c}_${numLevels}.log"
LOGFILE="./cosserat_continuum_${deformationOrder}_${rotationOrder}_${numLevels}.log"

#echo $RESULTPATH

# Set up directory where to store the results
# if ! test -d "$RESULTPATH"; then
#     mkdir $RESULTPATH
# fi
#rm $RESULTPATH/*

#################################################
#  run the actual simulation
#################################################

../build-cmake/src/cosserat-continuum-${deformationOrder}-${rotationOrder} cosserat-continuum-cantilever.parset -numLevels ${numLevels} | tee ${LOGFILE}

}


MAXPROCS=1


for numLevels in 1 2 3 4; do

  for order in 3; do

    # Do one simulation run
    #runComputation $numLevels $order $order

    #runComputation $numLevels 2 1

    runComputation $numLevels 3 2

    # Never have more than MAXPROCS processes
    NPROC=$(($NPROC+1))
    if [ "$NPROC" -ge "$MAXPROCS" ]; then
      wait
      NPROC=0
    fi

done

done
