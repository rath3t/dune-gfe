import subprocess

for order in range(1,4):

    minLevel =  5 + 1 - order;
    maxLevel = 10 + 1 - order;

    processList = []

    for numLevels in range(minLevel,maxLevel+1):
        subprocess.call(["echo", "foo"])

        LOGFILE = "./harmonicmaps_" + str(order) + "_" + str(numLevels) + ".log"

        ##  run the actual simulation
        executable = "./harmonicmaps-" + str(order)
        #p = subprocess.Popen([executable, "harmonicmaps-skyrmions-hexagon.parset", "-numLevels", str(numLevels)])
        p = subprocess.Popen(executable + " harmonicmaps-skyrmions-hexagon.parset -numLevels " + str(numLevels) + " | tee " + LOGFILE, shell=True)
        processList.append(p)

    # Wait for all simulation subprocesses before proceeding to the error measurement step
    exit_codes = [p.wait() for p in processList]

    subprocess.call(["echo", "Now measuring errors"])


    for numLevels in range(minLevel,maxLevel+1):
        # Measure the discretization errors against the solution on the finest grid
        LOGFILE = "./compute-disc-error_" + str(order) + "_" + str(numLevels) + ".log"

        #subprocess.Popen(["../build-cmake/src/compute-disc-error", "compute-disc-error-skyrmions-hexagon.parset",
                                        #"-order", str(order),
                                        #"-numLevels", str(numLevels),
                                        #"-numReferenceLevels", str(maxLevel),
                                        #"-simulationData", "harmonicmaps-result-" + str(order) + "-" + str(numLevels) + ".data",
                                        #"-referenceData", "harmonicmaps-result-" + str(order) + "-" + str(maxLevel) + ".data"])
        subprocess.Popen("../build-cmake/src/compute-disc-error compute-disc-error-skyrmions-hexagon.parset"
                                      + " -order " + str(order)
                                      + " -numLevels " + str(numLevels)
                                      + " -numReferenceLevels " + str(maxLevel)
                                      + " -simulationData harmonicmaps-result-" + str(order) + "-" + str(numLevels) + ".data"
                                      + " -referenceData harmonicmaps-result-" + str(order) + "-" + str(maxLevel) + ".data"
                                      + " | tee " + LOGFILE, shell=True)


