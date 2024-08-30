# Path to the example grid files in dune-grid
add_definitions(-DDUNE_GRID_EXAMPLE_GRIDS_PATH=\"${DUNE_GRID_EXAMPLE_GRIDS_PATH}\")

set(TESTS
  averagedistanceassemblertest
  cosseratrodenergytest
  cosseratrodtest
  embeddedglobalgfefunctiontest
  interpolationderivativestest
  localadolcstiffnesstest
  localgeodesicfefunctiontest
  localgfetestfunctiontest
  localintegralstiffnesstest
  localprojectedfefunctiontest
  orthogonalmatrixtest
  planarcosseratshelltest
  polardecompositiontest
  rotationtest
  surfacecosseratstressassemblertest
  svdtest
  targetspacetest
)

dune_add_test(NAME "filmonsubstratetest"
              SOURCES filmonsubstratetest.cc
              COMPILE_DEFINITIONS "MIXED_SPACE=0"
              TIMEOUT 2400)

dune_add_test(NAME "filmonsubstratetest-mixed"
              SOURCES filmonsubstratetest.cc
              COMPILE_DEFINITIONS "MIXED_SPACE=1"
              TIMEOUT 2400)

if (dune-curvedgrid_FOUND)
  set(TESTS test-cylinder ${TESTS})
endif()

if(dune-foamgrid_FOUND)
  set(TESTS nonplanarcosseratenergytest ${TESTS})
endif()

foreach(_test ${TESTS})
  dune_add_test(SOURCES ${_test}.cc)
  target_compile_options(${_test} PRIVATE -g)
endforeach()

dune_add_test(NAME harmonicmaptest-geodesic
              SOURCES harmonicmaptest.cc
              COMPILE_DEFINITIONS "GEODESICINTERPOLATION=1; CONFORMING=1"
              TIMEOUT 1200)

dune_add_test(NAME harmonicmaptest-projected
              SOURCES harmonicmaptest.cc
              COMPILE_DEFINITIONS "GEODESICINTERPOLATION=0; CONFORMING=1"
              TIMEOUT 1200)

# Run distributed tests
dune_add_test(NAME harmonicmaptest-projected-nonconforming
              SOURCES harmonicmaptest.cc
              COMPILE_DEFINITIONS "GEODESICINTERPOLATION=0; CONFORMING=0"
              MPI_RANKS 1 4
              TIMEOUT 1200
              CMAKE_GUARD MPI_FOUND)

dune_add_test(SOURCES cosseratcontinuumtest.cc
              TIMEOUT 1200)

dune_add_test(SOURCES geodesicfeassemblerwrappertest.cc
              MPI_RANKS 1 4
              TIMEOUT 600
              CMAKE_GUARD MPI_FOUND)

dune_add_test(SOURCES mixedriemannianpnsolvertest.cc
              MPI_RANKS 1 4
              TIMEOUT 600
              CMAKE_GUARD MPI_FOUND)

# Copy the example grid used for testing into the build dir
file(COPY grids/irregular-square.msh DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/grids)

# Copy an example for the stress plot into the build dir
file(COPY stressPlotData/stressPlotTestDeformation DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/stressPlotData)
file(COPY stressPlotData/stressPlotTestRotation DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/stressPlotData)
file(COPY stressPlotData/stressPlotTestInitialDeformation DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/stressPlotData)

# Copy an .adolcrc file into the build dir to make sure the tapes stay in memory and do not get written to disk because this causes errors sometimes
file(COPY .adolcrc DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
