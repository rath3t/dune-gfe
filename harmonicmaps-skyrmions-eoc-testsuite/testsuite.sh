#!/bin/bash

set -e

runComputation(){

numLevels=$1
order=$2

LOGFILE="./harmonicmaps_${order}_${numLevels}.log"

#  run the actual simulation
./harmonicmaps-${order} harmonicmaps-skyrmions-hexagon.parset -numLevels ${numLevels} | tee ${LOGFILE}

}


MAXPROCS=1

order=1
numReferenceLevels=10

for numLevels in $(seq 5 ${numReferenceLevels}); do

  # Do one simulation run
  runComputation $numLevels ${order}

  # Measure the discretization errors against the solution on the finest grid
  LOGFILE="./compute-disc-error_${order}_${numLevels}.log"

  ../build-cmake/src/compute-disc-error compute-disc-error-skyrmions-hexagon.parset \
                                        -order ${order} \
                                        -numLevels ${numLevels} \
                                        -numReferenceLevels ${numReferenceLevels} \
                                        -simulationData harmonicmaps-result-${order}-${numLevels}.data \
                                        -referenceData  harmonicmaps-result-${order}-${numReferenceLevels}.data | tee ${LOGFILE}
done

order=2
numReferenceLevels=9

for numLevels in $(seq 4 ${numReferenceLevels}); do

  # Do one simulation run
  runComputation $numLevels $order

  # Measure the discretization errors against the solution on the finest grid
  LOGFILE="./compute-disc-error_${order}_${numLevels}.log"

  ../build-cmake/src/compute-disc-error compute-disc-error-skyrmions-hexagon.parset \
                                        -order ${order} \
                                        -numLevels ${numLevels} \
                                        -numReferenceLevels ${numReferenceLevels} \
                                        -simulationData harmonicmaps-result-${order}-${numLevels}.data \
                                        -referenceData  harmonicmaps-result-${order}-${numReferenceLevels}.data | tee ${LOGFILE}
done

order=3
numReferenceLevels=8

for numLevels in $(seq 3 $numReferenceLevels); do

  # Do one simulation run
  runComputation $numLevels $order

  # Measure the discretization errors against the solution on the finest grid
  LOGFILE="./compute-disc-error_${order}_${numLevels}.log"

  ../build-cmake/src/compute-disc-error compute-disc-error-skyrmions-hexagon.parset \
                                        -order ${order} \
                                        -numLevels ${numLevels} \
                                        -numReferenceLevels ${numReferenceLevels} \
                                        -simulationData harmonicmaps-result-${order}-${numLevels}.data \
                                        -referenceData  harmonicmaps-result-${order}-${numReferenceLevels}.data | tee ${LOGFILE}
done

