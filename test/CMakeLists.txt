set(TESTS
  adolctest
  averagedistanceassemblertest
  cosseratenergytest
  cosseratrodenergytest
  cosseratrodtest
  embeddedglobalgfefunctiontest
  localgeodesicfefunctiontest
  localgfetestfunctiontest
  localprojectedfefunctiontest
  orthogonalmatrixtest
  polardecompositiontest
  rotationtest
  surfacecosseratstressassemblertest
  svdtest
  targetspacetest
)

dune_add_test(NAME "filmonsubstratetest"
              SOURCES filmonsubstratetest.cc
              COMPILE_DEFINITIONS "MIXED_SPACE=0"
              TIMEOUT 1200)

dune_add_test(NAME "filmonsubstratetest-mixed"
              SOURCES filmonsubstratetest.cc
              COMPILE_DEFINITIONS "MIXED_SPACE=1"
              TIMEOUT 1200)

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

# Run distributed tests
dune_add_test(SOURCES harmonicmaptest.cc
              MPI_RANKS 1 4
              TIMEOUT 1200
              CMAKE_GUARD MPI_FOUND)

dune_add_test(SOURCES cosseratcontinuumtest.cc
              TIMEOUT 1200)

dune_add_test(SOURCES geodesicfeassemblerwrappertest.cc
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
