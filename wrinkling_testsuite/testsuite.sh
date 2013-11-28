#!/bin/bash

set -e

runComputation(){

numLevels=$1
mu_c=$2
L_c=$3

RESULTPATH=`pwd`/cosserat_wrinkling_${mu_c}_${L_c}_${numLevels}/
LOGFILE="${RESULTPATH}/cosserat_wrinkling_${mu_c}_${L_c}_${numLevels}.log"

#echo $RESULTPATH

# Set up directory where to store the results
if ! test -d "$RESULTPATH"; then
    mkdir $RESULTPATH
fi
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

../cosserat-continuum -numLevels ${numLevels} -materialParameters.L_c ${L_c} -resultPath ${RESULTPATH} | tee ${LOGFILE}

}


MAXPROCS=4

mu_c=0

for numLevels in 1 2 3 4 5 6; do

    for L_c in 0.1 0.01 0.001 0.0001 1e-5 1e-6 1e-7 1e-8; do
 
        # Do one simulation run
        runComputation $numLevels $mu_c $L_c  &

        # Never have more than MAXPROCS processes
        NPROC=$(($NPROC+1))  
        if [ "$NPROC" -ge "$MAXPROCS" ]; then  
            wait  
            NPROC=0  
        fi  
      
    done

done
