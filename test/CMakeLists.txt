set(TESTS
  adolctest
  averagedistanceassemblertest
  cosseratenergytest
  cosseratrodenergytest
  harmonicenergytest
  localgeodesicfefunctiontest
  localgfetestfunctiontest
  localprojectedfefunctiontest
  orthogonalmatrixtest
  rigidbodymotiontest
  rotationtest
  surfacecosseratstressassemblertest
  svdtest
  targetspacetest
)

foreach(_test ${TESTS})
  dune_add_test(SOURCES ${_test}.cc)
  target_compile_options(${_test} PRIVATE -g)
endforeach()

# Run distributed tests
dune_add_test(SOURCES harmonicmaptest.cc
              MPI_RANKS 1 4
              TIMEOUT 600
              CMAKE_GUARD MPI_FOUND)

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
