#!/bin/bash

set -e

runComputation(){

numLevels=$1
L_c=$2

#RESULTPATH=`pwd`/richards_surfacewater_results_${leakage}_${richardsonDamping}/
LOGFILE="./cosserat_continuum_${L_c}_${numLevels}.log"

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

../cosserat-continuum -numLevels ${numLevels} -materialParameters.L_c ${L_c} | tee ${LOGFILE}

}


MAXPROCS=4


for numLevels in 1 2; do

    for L_c in 0.5 0.25; do     
 
        # Do one simulation run
        runComputation $numLevels $L_c  &

        # Never have more than MAXPROCS processes
        NPROC=$(($NPROC+1))  
        if [ "$NPROC" -ge "$MAXPROCS" ]; then  
            wait  
            NPROC=0  
        fi  
      
    done

done
