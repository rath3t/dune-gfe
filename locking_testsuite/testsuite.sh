#!/bin/bash

set -e

runComputation(){

numLevels=$1
mu_c=$2
kappa=$3

#RESULTPATH=`pwd`/richards_surfacewater_results_${leakage}_${richardsonDamping}/
#LOGFILE=${RESULTPATH}/"richards_${mu_c}_${numLevels}.log"
LOGFILE="./cosserat_continuum_${mu_c}_${numLevels}_${kappa}.log"

#echo $RESULTPATH

# Set up directory where to store the results
# if ! test -d "$RESULTPATH"; then
#     mkdir $RESULTPATH
# fi
#rm $RESULTPATH/*

#################################################
#  Make directories for the iterates
#################################################
# for i in $(eval echo "{0..$LASTTIMESTEP}"); do
#     ITERATESDIRNAME=${RESULTPATH}/iterates_$i
#     if ! test -d ${ITERATESDIRNAME}; then
#         mkdir ${ITERATESDIRNAME}
#     fi
# done

#################################################
#  run the actual simulation
#################################################

../cosserat-continuum -numLevels ${numLevels} -materialParameters.mu_c ${mu_c} -materialParameters.kappa ${kappa} | tee ${LOGFILE}


#################################################
#  Delete the directories for the iterates.
#  If we let the stay they eat up to much memory
#################################################

}


MAXPROCS=4


for numLevels in 1 2 3 4 5; do

    #for mu_c in 3.8462e+05 0; do
    mu_c=0

    for kappa in 1 0.1 0.01; do     
 
    # Do one simulation run
      runComputation $numLevels $mu_c $kappa &

      # Never have more than MAXPROCS processes
      NPROC=$(($NPROC+1))  
      if [ "$NPROC" -ge "$MAXPROCS" ]; then  
          wait  
          NPROC=0  
      fi  
      
    done

done
